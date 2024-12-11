import pandas as pd
import matplotlib.pyplot as plt

class Chart_Plot:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%Y')
        except Exception as e:
            print(f"Error loading data: {e}")

    
    def plot(self):
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

        plt.figure(figsize=(12, 8))
        for column in numeric_columns:
            plt.plot(self.data['Date'], self.data[column], label=column)

        plt.title("All Columns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("historicalData_Plot.png")
        print("Plot saved as image")

plotter = Chart_Plot("historicalData_IE00B5BMR087_clean.csv")
plotter.load_data()
plotter.plot()