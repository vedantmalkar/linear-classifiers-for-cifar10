import numpy as np
from softmax import softmax_loss
from svm import svm_loss

def gradient_check(loss_fn, W, X, y, reg, num_of_checks=10, h=0.00001):
    print(f"Gradient Checker Initialized")

    loss, grad_analytic = loss_fn(W, X, y, reg)
    print(f"Loss: {loss:.6f}")

    print(f"Checking from {num_of_checks} random weights")
    print("\n-----------------------------------------------------------------------------------------------")
    print("Index                  Numerical                Analytic                        Relative Error")
    print("-----------------------------------------------------------------------------------------------")

    for i in range(num_of_checks):
        ix = tuple([np.random.randint(3073), np.random.randint(10)])            

        grad_analytic_val = grad_analytic[ix]

        old_value = W[ix]
        
        W[ix] = old_value + h
        loss_plus, _ = loss_fn(W, X, y, reg)
        
        W[ix] = old_value - h
        loss_minus, _ = loss_fn(W, X, y, reg)
        
        W[ix] = old_value 
        
        grad_numerical_val = (loss_plus - loss_minus) / (2 * h)               # manually calculating grad. 

        numerator = abs(grad_numerical_val - grad_analytic_val)                 #relative error
        denominator = max(abs(grad_numerical_val), abs(grad_analytic_val))
        
        if denominator == 0:
            rel_error = 0.0
        else:
            rel_error = numerator / denominator
        
        print(f"{str(ix)}               {grad_numerical_val}              {grad_analytic_val}              {rel_error}")
    


def main():
    from data_utils import load_CIFAR_10, preprocess_data
    
    print("Initialized CIFAR data set")
    X_train, y_train, _, _ = load_CIFAR_10('cifar-10-batches-py')
    X_train, y_train, _, _ = preprocess_data(X_train, y_train, X_train[:10], y_train[:10])

    X_small = X_train[:20]
    y_small = y_train[:20]
    
    print(f"Using {X_small.shape[0]} examples for gradient check")

    W = np.random.randn(3073, 10) * 0.001
    reg = 0.0 

    print("-----------------------------------------------------------------")
    print("CHECKING SOFTMAX GRADIENT")
    print("-----------------------------------------------------------------")
    gradient_check(softmax_loss, W, X_small, y_small, reg, num_of_checks=10)

    print("\n")
    print("-----------------------------------------------------------------")
    print("CHECKING SVM GRADIENT")
    print("-----------------------------------------------------------------")
    gradient_check(svm_loss, W, X_small, y_small, reg, num_of_checks=10)
    
    print("\n")
    print("Gradient check complete!")


if __name__ == '__main__':
    main()