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





