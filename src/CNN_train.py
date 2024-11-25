import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from CNN_preprocessor import CNNDataPreprocessor
from stockprice_CNN import StockPriceCNN

class CNNTrainer:
    def __init__(self, data, target_column, time_steps=5, test_size=0.2, batch_size=32, learning_rate=0.001, num_epochs=100):
        self.data = data
        self.target_column = target_column
        self.time_steps = time_steps
        self.test_size = test_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use cuda (gpu) if possible otherwise use cpu
        
        # Preprocess data and split into training and validation sets
        self._prepare_data()
        
        # Initialize model, loss function, and optimizer
        self.model = StockPriceCNN(num_classes=1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _prepare_data(self):
        print("Preparing data...")

        """Prepare training and validation data for the CNN."""
        # Data preprocessing
        preprocessor = CNNDataPreprocessor(self.data, self.time_steps, self.target_column)
        X, y = preprocessor.process()  # X is image data, y is the target labels
        X = torch.stack(X) # Add the batch dimension to image -> needed for cnn training
        
        print("X.shape:", X.shape, "\nX:", X[:5], "\ny:", y)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        
        # Convert to PyTorch tensors and create DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                X_train,
                torch.tensor(y_train, dtype=torch.float32)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    def train(self):
        print("Training model...")

        """Train the CNN model."""
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)  # CNN processes image data
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
        print("Validating model...")

        """Validate the CNN model and calculate loss and MAPE."""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(self.X_val_tensor).squeeze()
            val_loss = self.criterion(val_outputs, self.y_val_tensor).item()
            val_mape = self.calculate_mape(self.y_val_tensor, val_outputs)
        return val_loss, val_mape.item()
    
    def test(self, test_data):
        print("Testing model...")

        """Test the CNN model on unseen data and calculate MAPE."""
        preprocessor = CNNDataPreprocessor(test_data, self.time_steps, self.target_column)
        X_test, y_test = preprocessor.process()
        X_test = torch.stack(X_test)
        
        # Convert to tensors and move to device
        X_test = X_test.to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test).squeeze()
            mape = self.calculate_mape(y_test, y_pred)
        return mape.item()
