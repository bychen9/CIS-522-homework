import numpy as np


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

        self.w = np.zeros(X.shape[1])

        for epoch in range(epochs):
            y_pred = self.predict(X)
            w_grad = np.mean(np.dot(y_pred - y, self.w))
            b_grad = y_pred - y

            self.w -= lr * w_grad
            self.b -= lr * b_grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b
