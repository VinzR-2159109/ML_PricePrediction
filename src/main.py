#!/usr/bin/env python3
"""
Module for training and testing a CNN on stock market data using images.
"""
import pandas as pd
import argparse
from CNN_train import CNNTrainer

__author__ = "Vinz Roosen, Jan-Mathijs Pex, Lars Gielen"
__version__ = "0.1.0"
__license__ = "GPLv3"

def main(args):
    """Main entry point of the application."""
    training_file = args.training_file
    testing_file = args.testing_file

    # Load training and testing data
    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv(testing_file)

    # Initialize the trainer with training data
    trainer = CNNTrainer(
        data=training_data,
        target_column='Last Close',  # Ensure the target column matches your dataset
        time_steps=5,
        test_size=0.2,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=25
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Test the model on the testing data and print MAPE
    print("Evaluating model on testing data...")
    mape = trainer.test(testing_data)
    print(f"Mean Absolute Percentage Error (MAPE) on testing data: {mape:.2f}%")

if __name__ == "__main__":
    """This is executed when run from the command line."""
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("training_file", help="Path to the training data file (CSV format).")
    parser.add_argument("testing_file", help="Path to the testing data file (CSV format).")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__)
    )

    args = parser.parse_args()
    main(args)
