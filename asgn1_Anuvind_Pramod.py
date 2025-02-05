import numpy as np
import json

class LinReg:
    def __init__(self, data):
        self.x = np.array([d[0] for d in data])
        self.y = np.array([d[1] for d in data])
        self.w = 0
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

    def fit(self, learning_rate: float = 0.01, n_iters: int = 200) -> None:
        for _ in range(n_iters):
            y_pred = self.forward(self.x)
            grad = self.gradient(self.x, self.y, y_pred)
            self.w -= learning_rate * grad




'''class LinReg:
    def __init__(self, data):
        """
        Data is expected to be in the form [(x_1,y_1), (x_2, y_2), ..., (x_n, y_n)]
        """
        self.data = np.array(data, dtype=np.float64)  # Convert list to numpy array
        self.w = np.random.randn(1)  # Initialize weight randomly
    
    def get_weights(self) -> np.ndarray:
        """Returns the current weight as a numpy array."""
        return self.w
    
    def forward(self, x):
        """Computes the predicted output: y_pred = w * x"""
        return self.w * x
    
    @staticmethod
    def loss(y, y_pred) -> float:
        """Computes the Mean Squared Error (MSE) loss."""
        return np.mean((y - y_pred) ** 2)
    
    @staticmethod
    def gradient(x, y, y_pred) -> float:
        """Computes the gradient of the loss with respect to the weights."""
        return -2 * np.mean(x * (y - y_pred))
    
    def fit(self, learning_rate: float = 0.01, n_iters: int = 20) -> None:
        """Performs gradient descent to update the weight."""
        for _ in range(n_iters):
            x_vals = self.data[:, 0]  # Extract x values
            y_vals = self.data[:, 1]  # Extract y values
            y_pred = self.forward(x_vals)
            grad = self.gradient(x_vals, y_vals, y_pred)
            self.w -= learning_rate * grad  # Update weight
'''
# Load JSON data
with open("asgn1_data_publish.json", "r") as f:
    dataset = json.load(f)

# Flatten the dataset (since it's stored in multiple lists)
flat_data = [item for sublist in dataset for item in sublist]

# Train the model
model = LinReg(flat_data)
model.fit()

# Print final weight
print("Learned Weight:", model.get_weights()[0])





