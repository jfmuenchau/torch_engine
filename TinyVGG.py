import torch
from torch import nn
class TinyVGG(nn.Module):
    """Creates TinyVGG model from CNN explainer website
    * input_shape: Number of colour channels
    * output_shape: Number of output classes
    * hidden_units: Number of hidden units. Standard is 10."""

    def __init__(self, input_shape, output_shape, hidden_units = 10):
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2))

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape))

    def forward(self, x):
        return self.classifier(self.convblock2(self.convblock1(x)))
