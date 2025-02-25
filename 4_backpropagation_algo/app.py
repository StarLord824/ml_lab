import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for the hidden layer and output layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size)  # Weights for the hidden layer
        self.b1 = np.zeros((1, self.hidden_size))  # Biases for the hidden layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size)  # Weights for the output layer
        self.b2 = np.zeros((1, self.output_size))  # Biases for the output layer

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative (used in backpropagation)
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward pass
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1  # Input to hidden layer
        self.a1 = self.sigmoid(self.z1)  # Output from hidden layer (activation)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Input to output layer
        self.a2 = self.sigmoid(self.z2)  # Output from the network (prediction)
        return self.a2

    # Backward pass (Backpropagation)
    def backward(self, X, y, learning_rate):
        # Calculate the error
        error = self.a2 - y

        # Output layer gradients
        d_a2 = error * self.sigmoid_derivative(self.a2)
        d_W2 = np.dot(self.a1.T, d_a2)
        d_b2 = np.sum(d_a2, axis=0, keepdims=True)

        # Hidden layer gradients
        d_a1 = np.dot(d_a2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2

    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Perform forward pass
            self.forward(X)
            
            # Perform backward pass and update weights
            self.backward(X, y, learning_rate)

            # Print the loss (mean squared error) for every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = np.mean(np.square(y - self.a2))  # MSE
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # Predict using the trained model
    def predict(self, X):
        return self.forward(X)


# Sample binary dataset (e.g., XOR problem)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR labels

# Initialize the neural network
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Train the neural network
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Test the neural network on the same input
predictions = nn.predict(X)
print("\nPredictions on the training data:")
print(predictions)

# Threshold predictions to make them binary
predictions = (predictions > 0.5).astype(int)
print("\nThresholded Predictions (binary):")
print(predictions)
