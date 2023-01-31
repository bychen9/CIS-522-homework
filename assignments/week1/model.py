import numpy as np
import torch


class LinearRegression:
    """
    A linear regression model that uses the closed-form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Use the closed form solution for linear regression to fit the model based on the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input data.

        Returns:
            None
        """
        X1 = np.vstack((np.ones(X.shape[0]), X.T)).T
        a = np.matmul(np.matmul(np.linalg.inv(np.matmul(X1.T, X1)), X1.T), y)
        self.b = a[0]
        self.w = a[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
           np.ndarray: The predicted output.
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Use gradient descent to fit the model based on the given input.

        Arguments:
            X (np.ndarray): input data
            y (np.ndarray): input data
            lr (float): learning rate
            epochs (int): number of epochs for gradient descent

        Returns:
            None
        """

        X = torch.tensor(X, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.double)

        w = torch.tensor(np.zeros(X.shape[1]), requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)

        for epoch in range(epochs):
            y_pred = X @ w + b
            loss = torch.mean((y_pred - y) ** 2)

            loss.backward()

            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()

        self.w = w.detach().numpy()
        self.b = b.detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b
