#!/usr/bin/env python3
import pandas as pd
import argparse
import optuna
import logging
from LSTM_train import LSTMTrainer
import matplotlib.pyplot as plt

__author__ = "Vinz Roosen, Jan-Mathijs Pex, Lars Gielen"
__version__ = "0.1.0"
__license__ = "GPLv3"

def setup_logging(output_file):
    logging.basicConfig(
        filename=output_file,
        filemode='w',
        format='%(message)s',
        level=logging.INFO
    )

def objective(trial, training_data, testing_data):
    time_steps = trial.suggest_int("time_steps", 1, 20)
    hidden_size = trial.suggest_int("hidden_size", 10, 100)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 50, 500)

    trainer = LSTMTrainer(
        data=training_data,
        target_column='Last Close',
        time_steps=time_steps,
        test_size=0.2,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    trainer.train()
    mape = trainer.test(testing_data)
    logging.info(f"Trial {trial.number}: MAPE = {mape:.2f}%, Params = {trial.params}")
    return mape

def main(args):
    setup_logging(output_file="output_file.txt")
    logging.info("Starting script...")

    training_file = args.training_file
    testing_file = args.testing_file

    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv(testing_file)

    study = optuna.create_study(direction="minimize")
    logging.info("Starting hyperparameter optimization...")
    study.optimize(lambda trial: objective(trial, training_data, testing_data), n_trials=50)

    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.figure.savefig("hyperparameter_importances.png", dpi=300)

    best_trial = study.best_trial
    logging.info("Best Trial:")
    logging.info(f"  MAPE: {best_trial.value:.2f}%")
    logging.info(f"  Hyperparameters: {best_trial.params}")

    best_params = best_trial.params
    trainer = LSTMTrainer(
        data=training_data,
        target_column='Last Close',
        time_steps=best_params["time_steps"],
        test_size=0.2,
        batch_size=best_params["batch_size"],
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        learning_rate=best_params["learning_rate"],
        num_epochs=best_params["num_epochs"]
    )
    trainer.train()

    final_mape = trainer.test(testing_data)
    logging.info(f"Final MAPE with optimized parameters: {final_mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__)
    )
    args = parser.parse_args()
    main(args)
