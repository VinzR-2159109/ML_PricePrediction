import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None
        self.model = LinearRegression()

    def load_data(self):
        self.data = pd.read_csv(self.file_name)
        self.data['Date'] = pd.to_datetime(self.data['Date'])

    def prepare_data(self):
        X = self.data[['Open', 'High', 'Low']]
        y = self.data['Last Close']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, validation_file):
        validation_data = pd.read_csv(validation_file)
        validation_data['Date'] = pd.to_datetime(validation_data['Date'])
        
        X_validation = validation_data[['Open', 'High', 'Low']]
        y_validation = validation_data['Last Close']
        
        y_pred = self.model.predict(X_validation)
        
        mape = np.mean(np.abs((y_validation - y_pred) / y_validation)) * 100
        
        return y_pred, y_validation, mape


    def visualize_results(self, y_test, y_pred, output_file="results_plot.png"):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
        plt.xlabel("Actual Last Close")
        plt.ylabel("Predicted Last Close")
        plt.title("Actual vs Predicted Values")
        plt.savefig(output_file)
        print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    file_name = "historicalData_IE00B5BMR087_clean.csv"
    validation_file = "october.csv"
    lr_model = LinearRegressionModel(file_name)

    lr_model.load_data()
    X_train, X_test, y_train, y_test = lr_model.prepare_data()
    
    lr_model.train_model(X_train, y_train)

    # Evaluate the model using the validation dataset
    y_pred, y_validation, mape = lr_model.evaluate_model(validation_file)

    print("Validation Performance on October Data:")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    lr_model.visualize_results(y_validation, y_pred, output_file="october_results_plot.png")

