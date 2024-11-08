#!/usr/bin/env python3
"""
Module Docstring
"""
import pandas as pd
import argparse
from LSTM_train import LSTMTrainer

__author__ = "Vinz Roosen, Jan-Mathijs Pex, Lars Gielen"
__version__ = "0.1.0"
__license__ = "GPLv3"

def main(args):
    """ Main entry point of the app """
    training_file = args.training_file
    testing_file = args.testing_file

    # Load training and testing data
    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv(testing_file)

    # Initialize the trainer with training data
    trainer = LSTMTrainer(
        data=training_data,
        target_column='Last Close',  # Adjust this if your target column name is different
        time_steps=5,
        test_size=0.2,
        batch_size=32,
        hidden_size=50,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=100
    )

    # Train the model on the training data
    trainer.train()

    # Evaluate the model on the testing data and calculate MAPE
    mape = trainer.test(testing_data)  # The `test` method in LSTMTrainer should calculate and return MAPE

    print("MAPE: {:.2f}%".format(mape))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__)
    )

    args = parser.parse_args()
    main(args)
