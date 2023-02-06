from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """
    Creates MLP module and performs forward pass.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.linear1 = nn.Linear(input_size, hidden_size)
        initializer(self.linear1.weight)
        self.layers.append(self.linear1)
        self.actv = activation
        if hidden_count > 1:
            for i in range(hidden_count - 1):
                linear_i = nn.Linear(hidden_size, hidden_size)
                initializer(linear_i.weight)
                self.layers.append(linear_i)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.actv()(layer(x))
        return self.out(x)
