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
        self.data['Last_Close_Lag_1'] = self.data['Last Close'].shift(1)
        self.data['Last_Close_MA_5'] = self.data['Last Close'].rolling(window=5).mean()
        self.data['Last_Close_MA_10'] = self.data['Last Close'].rolling(window=10).mean()

        self.data['Volatility'] = self.data['High'] - self.data['Low']
        self.data['Percentage_Change'] = (self.data['Last Close'] - self.data['Open']) / self.data['Open']
        
        # Relative Strength Index (RSI)
        delta = self.data['Last Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        self.data.bfill(inplace=True)  
        self.data.ffill(inplace=True)  

    def scale_data(self):
        feature_columns = ['Open', 'High', 'Low', 'Last Close', 'Last_Close_Lag_1', 'Last_Close_MA_5', 'Last_Close_MA_10']
        self.data_scaled = self.scaler.fit_transform(self.data[feature_columns])
        return self.data_scaled

    def create_sequences(self):
        x, y = [], []
        target_idx = self.data.columns.get_loc(self.target_column)

        for i in range(self.time_steps, len(self.data_scaled)):
            x.append(self.data_scaled[i - self.time_steps:i, :])
            y.append(self.data_scaled[i, target_idx])
            
        self.x, self.y = np.array(x), np.array(y)
        return self.x, self.y

    def to_tensors(self):
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        return x_tensor, y_tensor

    def process(self):
        self.add_features()
        self.scale_data()
        self.create_sequences()
        return self.to_tensors()
