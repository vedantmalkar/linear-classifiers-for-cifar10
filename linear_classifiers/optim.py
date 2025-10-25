import numpy as np 

def sgd_step(W, dW, learning_rate):
    W = W - learning_rate * dW      #updates Weights based on gradient
    return W

def create_batches(X, Y, batch_size):
    N = X.shape[0]
    
    indices = np.random.permutation(N)  #random shuffling
    X_random = X[indices]
    Y_random = Y[indices]
    
    batches = []
    for i in range(0, N, batch_size):           #creating batches randomly in sets of batch_size
        X_batch = X_random[i:i+batch_size]
        y_batch = Y_random[i:i+batch_size]
        batches.append((X_batch, y_batch))
    
    return batches
