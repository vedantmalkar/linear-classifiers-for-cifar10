import numpy as np
import pickle
import os

def load_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        X = data_dict[b'data']       #shape is (10000, 3072)
        Y = data_dict[b'labels']     
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
    return X,Y

def load_CIFAR_10(root):
    temp_x, temp_y = [], []
    for i in range(1, 6):
        batch = os.path.join(root, f"data_batch_{i}")
        X, Y = load_batch(batch)
        temp_x.append(X) # 5 arrays of array(10000, 3, 32, 32)
        temp_y.append(Y) # 5 arrays of array(10000)
    X_train = np.concatenate(temp_x) # single array of 50k images (shape of (50000,3,32,32)))
    Y_train = np.concatenate(temp_y)
    X_test, Y_test = load_batch(os.path.join(root, "test_batch"))
    print("loaded CIFAR 10")
    return X_train, Y_train, X_test, Y_test

def preprocess_data(X_train, Y_train, X_test, Y_test):
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float64) #flatten
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float64)
    
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image   # highlights diffrence b/w imgs
    X_test -= mean_image
    
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])  #add bias (1) 
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    print("preprocessed CIFAR data")
    return X_train, Y_train, X_test, Y_test


