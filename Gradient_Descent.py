import numpy as np

N = 100
D = 2

N_per_class = N//2


X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:N_per_class,:] = X[:N_per_class,:] - 2 * np.ones((N_per_class,D))

# center the last 50 points at (2, 2)
X[N_per_class:,:] = X[N_per_class:,:] + 2 * np.ones((N_per_class,D))

# labels: first N_per_class are 0, last N_per_class are 1
T = np.array([0] * N_per_class + [1] * N_per_class)

# add a column of ones
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

# Define Cross-Entropy Error function
def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

# Run Gradient Descent
# First set up our learning rate
learning_rate = 0.1
for i in range(100):  # Suppose we run it 100 times
    if i % 10 == 0:  # print the Cross-Entropy Error function per 10 times
        print(cross_entropy(T, Y))
        
    # Update the weights
    # The original differntial should be Xb.T.dot(Y - T). 
    # Since we wnat to move the weights to the minimize point.
    # we need to add a negative on the slope.
    w += learning_rate * Xb.T.dot(T - Y)  
    
    # recalculate Y
    Y = sigmoid(Xb.dot(w))


print('final w: ', w)
