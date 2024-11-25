import torch
import torch.nn as nn
import torchvision.models as models

class StockPriceCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(StockPriceCNN, self).__init__()
        
        # Use ResNet18 as CNN-model
        self.base_model = models.resnet18(pretrained=False)
        
        # Replace last complete layer with modified layer
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
