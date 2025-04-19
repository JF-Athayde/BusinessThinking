import numpy as np
import tqdm
import os

# Converts the highest value in a prediction vector to 1, others to 0
def magic(predict):
    max_value = max(predict)
    result = []

    for i in predict:
        if i == max_value:
            result.append(1)
        else:
            result.append(0)
    
    return result

# Calculates absolute difference between two lists
def subtract(list1, list2):
    result = []
    for a, b in zip(list1, list2):
        result.append(abs(a - b))
    return result

# Reads a file and converts it into a list of lists of floats
def read_file_as_list(file_path, delimiter=","):
    list_of_lists = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                list_of_lists.append([float(i) for i in line.strip().split(delimiter)])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return list_of_lists

# Normalizes a list of numbers between 0 and 1
def normalize_data(data_list):
    min_val = min(data_list)
    max_val = max(data_list)
    
    if max_val == min_val:
        return [1] * len(data_list)
    
    return [(x - min_val) / (max_val - min_val) for x in data_list]

# Calculates Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

# Dummy optimizer (currently not used)
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_weights(self, weights, gradient):
        return weights - self.learning_rate * gradient

# Simple deep learning model with 3 layers and customizable activation
class DeepLearning:
    def __init__(self, X, Y, learning_rate=0.01, hidden_layers=None, activation='sigmoid'):
        self.X, self.Y = np.array(X), np.array(Y)
        self.hidden_layers = hidden_layers if hidden_layers else [10, 10]
        self.learning_rate = learning_rate
        self.activation = activation
        self._initialize_parameters()

    def _initialize_parameters(self):
        input_size, output_size = self.X.shape[1], self.Y.shape[1]
        self.w1, self.w2, self.w3 = (
            np.random.randn(input_size, self.hidden_layers[0]),
            np.random.randn(self.hidden_layers[0], self.hidden_layers[1]),
            np.random.randn(self.hidden_layers[1], output_size)
        )
        self.b1, self.b2, self.b3 = (
            np.zeros((1, self.hidden_layers[0])),
            np.zeros((1, self.hidden_layers[1])),
            np.zeros((1, output_size))
        )

    def activation_function(self, z):
        if self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z):
        if self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            sig = self.activation_function(z)
            return sig * (1 - sig)

    def forward_propagation(self):
        self.a1 = self.activation_function(np.dot(self.X, self.w1) + self.b1)
        self.a2 = self.activation_function(np.dot(self.a1, self.w2) + self.b2)
        self.a3 = self.activation_function(np.dot(self.a2, self.w3) + self.b3)

    def backward_propagation(self):
        m = self.Y.shape[0]
        dz3 = self.a3 - self.Y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = np.dot(dz3, self.w3.T) * self.activation_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.w2.T) * self.activation_derivative(self.a1)
        dW1 = np.dot(self.X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self._update_parameters(dW1, db1, dW2, db2, dW3, db3)

    def _update_parameters(self, dW1, db1, dW2, db2, dW3, db3):
        self.w1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.w3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

    def train(self, epochs=1000, verbose=True):
        for epoch in tqdm.tqdm(range(epochs)):
            self.forward_propagation()
            self.backward_propagation()

            if (epoch + 1) % 100 == 0 and verbose:
                loss = mean_squared_error(self.Y, self.a3)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        a1 = self.activation_function(np.dot(X, self.w1) + self.b1)
        a2 = self.activation_function(np.dot(a1, self.w2) + self.b2)
        a3 = self.activation_function(np.dot(a2, self.w3) + self.b3)
        return a3

# Load training data
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(current_directory, 'datas')
X_train_path = os.path.join(data_path, 'x_train.dll')
Y_train_path = os.path.join(data_path, 'y_train.dll')

X = read_file_as_list(X_train_path)
Y = read_file_as_list(Y_train_path)

X_train = []
Y_train = []

# Normalize input features
for x in X:
    X_train.append(normalize_data(x))

# Convert label to one-hot encoding
for y in Y:
    if y[0] == 4:
        Y_train.append([0, 0, 0, 0, 1])
    elif y[0] == 3:
        Y_train.append([0, 0, 0, 1, 0])
    elif y[0] == 2:
        Y_train.append([0, 0, 1, 0, 0])
    elif y[0] == 1:
        Y_train.append([0, 1, 0, 0, 0])
    elif y[0] == 0:
        Y_train.append([1, 0, 0, 0, 0])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Build and train the model
model = DeepLearning(X_train, Y_train, learning_rate=0.001, hidden_layers=[10, 10], activation='sigmoid')
model.train(epochs=1000000, verbose=False)

# Evaluate predictions
mse_list = []
mae_list = []
predictions = model.predict(X_train.tolist())
correct_predictions = 0

for predicted, true in zip(predictions.tolist(), Y_train.tolist()):
    mae_list.append(sum(subtract(true, predicted)) / len(subtract(true, predicted)))

    if true == magic(predicted):
        correct_predictions += 1

print(correct_predictions, round(100 - (sum(mae_list) / len(mae_list)) * 100, 3))
