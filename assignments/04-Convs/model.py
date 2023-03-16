import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    CNN Model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv2d1 = nn.Conv2d(num_channels, 10, 3)
        self.conv2d2 = nn.Conv2d(num_channels, 10, 3)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(10 * 15 * 15 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x1 = self.conv2d1(x)
        x1 = self.pool(x1)
        x1 = self.relu(x1)
        x1 = torch.flatten(x1, 1)

        x2 = self.conv2d2(x)
        x2 = self.pool(x2)
        x2 = self.relu(x2)
        x2 = torch.flatten(x2, 1)

        x = torch.cat((x1, x2), 1)
        x = self.fc(x)

        return x
