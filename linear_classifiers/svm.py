import numpy as np

def svm_loss(W,X,y,reg):
    dW = np.zeros_like(W)
    number_to_train = X.shape[0]
    number_of_classes = W.shape[1]
    loss = 0.0

    for i in range(number_to_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        num_violating_classes = 0
        
        for j in range(number_of_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1.0          # make margin only if not the right lable
            
            if margin > 0:                                          # update loss if margin for this lable is positive
                loss += margin
                num_violating_classes += 1
                dW[:, j] += X[i]
        
        dW[:, y[i]] -= num_violating_classes * X[i]
    
    loss /= number_to_train
    dW /= number_to_train
    
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    return loss, dW
