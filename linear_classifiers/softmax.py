import numpy as np 

def linear_score_creation(X,W):
    return X.dot(W)

def softmax_function(scores):
    scores_shifted = scores - np.max(scores,axis=1,keepdims=True)       #subtract the max value of each row of scores from the scores matrix
    exp_scores = np.exp(scores_shifted)                         #take exp to remove negative and large numbers
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)      #probability by dividing by sum 
    return probs

def cross_entropy_loss(probs,y):
    N = probs.shape[0]
    lowest = 1e-10
    probs = np.clip(probs, lowest, 1.0)                    #changes 0 probability to 10^-10
    correct_log_probs = -np.log(probs[np.arange(N), y])     #array of correct probailities only (negative log taken)
    loss = np.sum(correct_log_probs) / N                    # average loss
    return loss

def regularization_loss(W, reg):        #L2 regularization loss
    return reg * np.sum(W * W)

def compute_gradient(X, probs, y, reg, W):
    N = X.shape[0]
    probs_temp = probs.copy()
    probs_temp[np.arange(N), y] -= 1    #subtract 1 from the correct probabilities 
    dW = np.dot(X.T, probs_temp)              # (3073,50000) * (50000,10) = (3073,10) , the gradient for each feature for each class on how to change it  
    dW /= N  
    dW += 2 * reg * W                   # L2  
    return dW

def softmax_loss(W, X, y, reg):
    scores = linear_score_creation(X, W)
    probs = softmax_function(scores)
    data_loss = cross_entropy_loss(probs, y)
    reg_loss = regularization_loss(W, reg)
    loss = data_loss + reg_loss
    dW = compute_gradient(X, probs, y, reg, W)
    
    return loss, dW