import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class LSTMDataPreprocessor:
    def __init__(self, data, time_steps=5, target_column='Last Close'):
        self.data = data
        self.time_steps = time_steps
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.x = None
        self.y = None

    def add_features(self):
        ## REPORT: different types of features
        self.data['Last_Close_Lag_1'] = self.data['Last Close'].shift(1)
        self.data['Last_Close_MA_5'] = self.data['Last Close'].rolling(window=5).mean()
        self.data['Last_Close_MA_10'] = self.data['Last Close'].rolling(window=10).mean()
        
        ## REPORT: different types of filling the missing values
        self.data.bfill(inplace=True)  
        self.data.ffill(inplace=True)  

    def scale_data(self):
        ## REPORT: widely different ranges can cause the model to focus disproportionately on features with larger values.
        feature_columns = ['Open', 'High', 'Low', 'Last Close', 'Last_Close_Lag_1', 'Last_Close_MA_5', 'Last_Close_MA_10']
        self.data_scaled = self.scaler.fit_transform(self.data[feature_columns])
        return self.data_scaled

    def create_sequences(self):
        x, y = [], []
        target_idx = self.data.columns.get_loc(self.target_column)  # Get index of target column after scaling (should be 4)

        for i in range(self.time_steps, len(self.data_scaled)):
            x.append(self.data_scaled[i - self.time_steps:i, :])  # Past 'time_steps (5)' days as input
            y.append(self.data_scaled[i, target_idx])  # Target is the closing price of the last day in sequence
            
        self.x, self.y = np.array(x), np.array(y)
        return self.x, self.y

    def to_tensors(self):
        # Convert sequences to PyTorch tensors (REPORT: higher dimensional arrays)
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        return x_tensor, y_tensor

    def process(self):
        self.add_features()
        self.scale_data()
        self.create_sequences()
        return self.to_tensors()
