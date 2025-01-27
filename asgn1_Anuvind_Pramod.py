import numpy as np

class LinReg:
    def __init__(self, data):
        self.x = np.array([d[0] for d in data])
        self.y = np.array([d[1] for d in data])
        self.w = 0.0  
    def get_weights(self):
        return np.array([self.w])

    def forward(self, x):
        return self.w * x

    @staticmethod
    def loss(y, y_pred):
        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def gradient(x, y, y_pred):
        n = x.shape[0]
        return (-2 / n) * np.sum((y - y_pred) * x)

    def fit(self, learning_rate: float = 0.05, n_iters: int = 50) -> None:
        for _ in range(n_iters):
            y_pred = self.forward(self.x)
            grad = self.gradient(self.x, self.y, y_pred)
            self.w -= learning_rate * grad



