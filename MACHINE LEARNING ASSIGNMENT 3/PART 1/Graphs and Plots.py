import numpy as np
import matplotlib.pyplot as plt

# Example Loss Values (Replace these with actual losses from your training)
epochs = 10000
sigmoid_losses = np.random.uniform(0.7, 0.0, epochs)  # Replace with actual Sigmoid losses
tanh_losses = np.random.uniform(0.6, 0.0, epochs)     # Replace with actual Tanh losses

# Plot Sigmoid Activation Loss Curve
def plot_sigmoid_loss():
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), sigmoid_losses, label='Sigmoid Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve - Sigmoid Activation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Tanh Activation Loss Curve
def plot_tanh_loss():
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), tanh_losses, label='Tanh Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve - Tanh Activation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Both Loss Curves for Comparison
def plot_combined_loss():
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), sigmoid_losses, label='Sigmoid Loss', color='blue')
    plt.plot(range(epochs), tanh_losses, label='Tanh Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Sigmoid vs Tanh Activation - Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call Plotting Functions
print("Plotting Sigmoid Loss Curve...")
plot_sigmoid_loss()

print("Plotting Tanh Loss Curve...")
plot_tanh_loss()

print("Plotting Combined Loss Curve for Comparison...")
plot_combined_loss()
