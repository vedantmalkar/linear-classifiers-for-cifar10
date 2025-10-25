import numpy as np
import argparse
from tqdm import tqdm
from data_utils import load_CIFAR_10, preprocess_data
from softmax import *
from optim import *
import matplotlib.pyplot as plt
from svm import *

def accuracy(W,X,y):
    scores = X.dot(W)
    predictions = np.argmax(scores, axis=1)
    return np.mean(predictions == y)            # finds mean of number of correct predictions 

def train(X_train, Y_train, X_val, Y_val, loss_function, learning_rate=0.001, reg=0.0001, num_epochs=20, batch_size=200):
    num_of_features = 3073
    num_of_classes = 10
    W = np.random.randn(num_of_features, num_of_classes) * 0.001
    
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        batches = create_batches(X_train, Y_train, batch_size)
        epoch_losses = []
        
        for X_batch, Y_batch in tqdm(batches, desc=f'Epoch {epoch+1}/{num_epochs}'):
            loss, dW = loss_function(W, X_batch, Y_batch, reg)
            W = sgd_step(W, dW, learning_rate)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        train_acc = accuracy(W, X_train, Y_train)      
        val_acc = accuracy(W, X_val, Y_val)             # validation sets used to check if model is not "memorizing" the traning set
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}')
    
    return W, loss_history, train_acc_history, val_acc_history

def plot_stats(loss_history, train_acc_history, val_acc_history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='train', marker='o')
    plt.plot(val_acc_history, label='val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='softmax', choices=['softmax', 'svm'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=200)
    args = parser.parse_args()
    
    print(f'Training {args.model} with learning rate={args.lr}, reg={args.reg}')
    
    X_train, y_train, X_test, y_test = load_CIFAR_10('cifar-10-batches-py')
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    num_val = 5000
    X_val, y_val = X_train[:num_val], y_train[:num_val]
    X_train, y_train = X_train[num_val:], y_train[num_val:]
    
    print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')
    
    if args.model == 'softmax':
        loss_fn = softmax_loss
    elif args.model == 'svm':
        loss_fn = svm_loss
    
    W, loss_hist, train_acc, val_acc = train(X_train, y_train, X_val, y_val, loss_fn,learning_rate=args.lr, reg=args.reg,num_epochs=args.epochs,batch_size=args.batch_size)
    
    test_acc = accuracy(W, X_test, y_test)
    print(f'\nTest accuracy: {test_acc:.3f}')
    
    plot_stats(loss_hist, train_acc, val_acc)

if __name__ == '__main__':
    main()