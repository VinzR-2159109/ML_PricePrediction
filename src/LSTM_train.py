import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

from LSTM_preprocessor import LSTMDataPreprocessor
from stockprice_LSTM import StockPriceLSTM

class LSTMTrainer:
    def __init__(self, data, target_column, time_steps=5, test_size=0.2, batch_size=32, hidden_size=50, num_layers=2, learning_rate=0.001, num_epochs=100):
        self.data = data
        self.target_column = target_column
        self.time_steps = time_steps
        self.test_size = test_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Preprocess data and split into training and validation sets
        self._prepare_data()
        
        # Initialize model, loss function, and optimizer
        self.model = StockPriceLSTM(self.input_size, hidden_size, num_layers, output_size=1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _prepare_data(self):
        # Data preprocessing
        preprocessor = LSTMDataPreprocessor(self.data, self.time_steps, self.target_column)
        X, y = preprocessor.process()
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        
        # Convert to PyTorch tensors and create DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.X_val_tensor = X_val.to(self.device)
        self.y_val_tensor = y_val.to(self.device)
        self.input_size = X_train.shape[2]  # Number of features

    def calculate_mape(self, y_true, y_pred):
        """ Calculate Mean Absolute Percentage Error (MAPE) """
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Calculate average training loss for the epoch
            train_loss /= len(self.train_loader)
            val_loss, val_mape = self.validate()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.2f}%')

    def validate(self):
        """ Calculate validation loss and MAPE """
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(self.X_val_tensor).squeeze()
            val_loss = self.criterion(val_outputs, self.y_val_tensor).item()
            val_mape = self.calculate_mape(self.y_val_tensor, val_outputs)
        return val_loss, val_mape.item()
    
    def test(self, test_data):
        """ Test the model on the test data and calculate MAPE """
        preprocessor = LSTMDataPreprocessor(test_data, self.time_steps, self.target_column)
        X_test, y_test = preprocessor.process()
        
        # Convert to tensors and move to device
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test).squeeze()
            mape = self.calculate_mape(y_test, y_pred)
        return mape.item()