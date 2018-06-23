import numpy as np
N = 100
D = 2

X = np.random.randn(N, D)

# center the first 50 points at (-2,-2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
# center the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# Create targets
T = np.array([0] * 50 + [1] * 50)

ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# Randomly initialize the weights
w = np.random.randn(D + 1)

# Calculate the model output
z = Xb.dot(w)

# Define Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

# Define Cross Entropy Error function
def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print(cross_entropy(T, Y))


# Suppose we've use the sigmoid function to calculate our weights and get the below answer.
# Let's see how well will it perform
w = np.array([0, 4, 4])
z = Xb.dot(w)
Y = sigmoid(z)
print(cross_entropy(T, Y))



