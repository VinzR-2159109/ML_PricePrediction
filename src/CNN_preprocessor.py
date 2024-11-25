import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
import torchvision.transforms as transforms

class CNNDataPreprocessor:
    def __init__(self, data, time_steps=5, target_column='Last Close', save_dir='charts'):
        self.data = data
        self.time_steps = time_steps
        self.target_column = target_column
        self.save_dir = save_dir
        self.X = []
        self.y = []

        # Directory to save charts if user wants to
        # os.makedirs(self.save_dir, exist_ok=True)

    def add_features(self):
        print("Adding features to data...")

        # Add extra features
        self.data['Last_Close_MA_5'] = self.data['Last Close'].rolling(window=5).mean()
        self.data['Last_Close_MA_10'] = self.data['Last Close'].rolling(window=10).mean()
        self.data.bfill(inplace=True)  # Vul ontbrekende waarden in
        self.data.ffill(inplace=True)

    def create_charts(self):
        print("Creating charts...")

        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Change resolution to something square -> needed for cnn
            transforms.ToTensor(),
        ])

        for i in range(self.time_steps, len(self.data)):
            subset = self.data.iloc[i - self.time_steps:i]
            
            # Make graph
            plt.figure(figsize=(5, 3))  # Afmetingen voor consistente output
            plt.plot(subset.index, subset['Last Close'], label='Last Close', color='blue')
            plt.plot(subset.index, subset['Last_Close_MA_5'], label='MA 5', color='orange')
            plt.plot(subset.index, subset['Last_Close_MA_10'], label='MA 10', color='green')
            plt.legend(loc='best')
            # plt.title(f"Stock Trends (Ending on {self.data.index[i]})")
            plt.title('')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True)

            # Render to in-memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='PNG')
            plt.close()
            buf.seek(0)

            # Load image with PILLOW and convert to tensor with transform made earlier
            image = Image.open(buf).convert("RGB")
            tensor = transform(image)

            # Save image to given folder
            # chart_path = os.path.join(self.save_dir, f"chart_{i}.png")
            # plt.savefig(chart_path)
            # plt.close()

            # Add image and value of target_column to X and y
            self.X.append(tensor)
            self.y.append(self.data.iloc[i][self.target_column])

            buf.close()

    def process(self):
        self.add_features()
        self.create_charts()
        return self.X, self.y
