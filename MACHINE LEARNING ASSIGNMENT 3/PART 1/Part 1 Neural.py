import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Base Layer Class
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

# Xavier Initialization Function
def xavier_init(size):
    return np.random.randn(*size) * np.sqrt(2 / sum(size))

# Linear Layer with Xavier Initialization
class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = xavier_init((input_size, output_size))
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

# Tanh Activation Layer
class TanhLayer(Layer):
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (1 - np.power(self.output, 2))

# Sigmoid Activation Layer
class SigmoidLayer(Layer):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.output * (1 - self.output))

# ReLU Activation Layer
class ReLULayer(Layer):
    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        relu_derivative = np.where(self.output > 0, 1, 0)
        return output_gradient * relu_derivative

# Mean Squared Error (MSE) Loss
class MSELoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean(np.power(predictions - targets, 2))

    def backward(self):
        return 2 * (self.predictions - self.targets) / self.targets.size

# Sequential Model Class
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
        weights = {f"layer_{i}_weights": layer.weights for i, layer in enumerate(self.layers) if isinstance(layer, LinearLayer)}
        biases = {f"layer_{i}_bias": layer.bias for i, layer in enumerate(self.layers) if isinstance(layer, LinearLayer)}
        np.savez(filename, **weights, **biases)

# XOR Inputs and Outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Helper function to train the model and store losses
def train_model(model, loss_function, epochs, learning_rate):
    losses = []
    for epoch in range(epochs):
        predictions = model.forward(X)
        loss = loss_function.forward(predictions, y)
        losses.append(loss)

        # Backpropagation
        loss_grad = loss_function.backward()
        model.backward(loss_grad, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return losses

# Train and Test with Tanh Activation
print("\nTraining with Tanh Activation:")
model_tanh = Sequential()
model_tanh.add(LinearLayer(2, 4))  # Hidden layer with 2 nodes
model_tanh.add(TanhLayer())
model_tanh.add(LinearLayer(4, 1))
model_tanh.add(SigmoidLayer())  # Output layer with Sigmoid activation

# Train the model with Tanh activation
tanh_losses = train_model(model_tanh, MSELoss(), epochs=30000, learning_rate=0.11)

# Save the trained weights for Tanh model
model_tanh.save_weights(r"E:\MACHINE LEARNING ASSIGNMENT 3\PART 1\XOR_tanh_solved.w")

# Test the trained model with Tanh Activation
tanh_predictions = model_tanh.forward(X)
print("Test Predictions (Tanh):")
print(tanh_predictions)

# Train and Test with Sigmoid Activation
print("\nTraining with Sigmoid Activation:")
model_sigmoid = Sequential()
model_sigmoid.add(LinearLayer(2, 4))  # Hidden layer with 2 nodes
model_sigmoid.add(SigmoidLayer())
model_sigmoid.add(LinearLayer(4, 1))
model_sigmoid.add(SigmoidLayer())  # Output layer with Sigmoid activation

# Train the model with Sigmoid activationa
sigmoid_losses = train_model(model_sigmoid, MSELoss(), epochs=30000, learning_rate=0.1)

# Save the trained weights for Sigmoid model
model_sigmoid.save_weights(r"E:\MACHINE LEARNING ASSIGNMENT 3\PART 1\XOR_sigmoid_solved.w")

# Test the trained model with Sigmoid Activation
sigmoid_predictions = model_sigmoid.forward(X)
print("Test Predictions (Sigmoid):")
print(sigmoid_predictions)

# Plot the loss curves for Tanh vs Sigmoid (adjusted learning rate)
plt.figure(figsize=(10, 6))
plt.plot(tanh_losses, label='Tanh Loss', color='green')
plt.plot(sigmoid_losses, label='Sigmoid Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Tanh vs Sigmoid Activation - Training Loss')
plt.legend()
plt.grid(True)
plt.show()
