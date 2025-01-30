import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
dataset = np.load(r"E:\MACHINE LEARNING ASSIGNMENT 3\PART 2\nyc_taxi_data.npy", allow_pickle=True).item()
X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]

# Convert to DataFrame and handle non-numeric columns
X_train = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').dropna(axis=1)
X_test = pd.DataFrame(X_test).apply(pd.to_numeric, errors='coerce').dropna(axis=1)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Log-transform the target values
y_train = np.log1p(y_train.to_numpy()).reshape(-1, 1)
y_val = np.log1p(y_val.to_numpy()).reshape(-1, 1)
y_test = np.log1p(y_test.to_numpy()).reshape(-1, 1)

# Define the base layer class
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

# Linear Layer with Xavier Initialization, L2 Regularization, and Gradient Clipping
class LinearLayer(Layer):
    def __init__(self, input_size, output_size, regularization_lambda=0.01):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))
        self.reg_lambda = regularization_lambda
        self.clip_value = 5.0  # Set clip value to prevent large gradient updates

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Compute gradients
        weights_gradient = np.dot(self.input.T, output_gradient) + self.reg_lambda * self.weights
        input_gradient = np.dot(output_gradient, self.weights.T)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Apply gradient clipping
        np.clip(weights_gradient, -self.clip_value, self.clip_value, out=weights_gradient)
        np.clip(bias_gradient, -self.clip_value, self.clip_value, out=bias_gradient)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

# Leaky ReLU Activation Layer
class LeakyReLULayer(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, input):
        self.output = np.where(input > 0, input, self.alpha * input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * np.where(self.output > 0, 1, self.alpha)

# Mean Squared Error (MSE) Loss
class MSELoss:
    def forward(self, predictions, targets):
        return np.mean(np.square(predictions - targets))

    def backward(self, predictions, targets):
        return 2 * (predictions - targets) / targets.size

# Define test accuracy function
def test_accuracy(y_pred, y_true, tolerance=0.1):
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(y_true)
    accuracy = np.mean(np.abs((y_pred - y_true) / y_true) <= tolerance) * 100
    return accuracy

# Define the sequential model class
class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def save_weights(self, filename):
        np.savez(filename, **{f"layer_{i}_weights": layer.weights for i, layer in enumerate(self.layers) if isinstance(layer, LinearLayer)})

# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def check(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# RMSLE metric calculation
def rmsle(y_pred, y_true):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# Train model function with smaller learning rate
def train_model(model, X_train, y_train, X_val, y_val, loss_function, epochs, learning_rate=0.1):
    train_losses, val_losses = [], []
    early_stopping = EarlyStopping()

    for epoch in range(epochs):
        predictions = model.forward(X_train)
        train_loss = loss_function.forward(predictions, y_train)
        train_losses.append(train_loss)

        gradient = loss_function.backward(predictions, y_train)
        model.backward(gradient, learning_rate)

        val_predictions = model.forward(X_val)
        val_loss = loss_function.forward(val_predictions, y_val)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Validation Loss = {val_loss}")

        if early_stopping.check(val_loss):
            print("Early stopping triggered.")
            break

    return train_losses, val_losses

# Build and train three different models
models = []
results = []

# Model 1: Shallow model
model1 = Sequential()
model1.add(LinearLayer(X_train.shape[1], 32))
model1.add(LeakyReLULayer())
model1.add(LinearLayer(32, 1))
models.append(model1)

# Model 2: Deeper model
model2 = Sequential()
model2.add(LinearLayer(X_train.shape[1], 64))
model2.add(LeakyReLULayer())
model2.add(LinearLayer(64, 32))
model2.add(LeakyReLULayer())
model2.add(LinearLayer(32, 1))
models.append(model2)

# Model 3: Wide model
model3 = Sequential()
model3.add(LinearLayer(X_train.shape[1], 128))
model3.add(LeakyReLULayer())
model3.add(LinearLayer(128, 1))
models.append(model3)

# Train all models and store results
for i, model in enumerate(models):
    print(f"\nTraining Model {i + 1}:")
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val, 
        loss_function=MSELoss(), epochs=30, learning_rate=0.1
    )
    test_predictions = model.forward(X_test)
    test_rmsle = rmsle(np.expm1(test_predictions), np.expm1(y_test))
    test_acc = test_accuracy(test_predictions, y_test)
    results.append((train_losses, val_losses, test_rmsle, test_acc))
    print(f"Model {i + 1} Test RMSLE: {test_rmsle}, Test Accuracy: {test_acc}%")

# Plot individual training vs validation loss for each model
def plot_individual_losses(results):
    for i, (train_losses, val_losses, _, _) in enumerate(results):
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label=f'Model {i + 1} - Train Loss')
        plt.plot(val_losses, label=f'Model {i + 1} - Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training vs Validation Loss for Model {i + 1}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot the individual losses for each model separately
plot_individual_losses(results)

# Compare Test RMSLE and Accuracy for all models
for i, (_, _, test_rmsle, test_acc) in enumerate(results):
    print(f"Model {i + 1} Test RMSLE: {test_rmsle}, Test Accuracy: {test_acc}%")
