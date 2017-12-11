import numpy as np

def load_dataset():
    dataset = np.genfromtxt('ex1data2.txt', delimiter=',') 
    X = dataset[:,:-1]   
    y = dataset[:, [-1]]
    X = np.insert(X, 0, 1, axis=1)
    
    return X, y

def feature_standardization(X):
    
    for col in range(1, X.shape[1]):
        mean = np.mean(X[:, col])
        sigma = np.std(X[:, col])
        X_ = (X[:, col] - mean) / sigma
        X[:, col] = X_
    
    return X

def calculate_theta(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def main():
    X, y = load_dataset()
    X = feature_standardization(X)
    theta = calculate_theta(X, y)
    print("Theta's: {}".format(theta))

if __name__ == "__main__":
    main()
