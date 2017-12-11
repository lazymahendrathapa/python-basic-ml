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

def calculate_output(X, theta):
    return X.dot(theta)

def calculate_cost(X,y, theta):
    m = y.size
    output = calculate_output(X, theta)
    return np.sum(np.square(output - y)) / (2 * m)

def gradient_descent(X, y, alpha, iteration_number):

    theta = np.random.random_sample([X.shape[1],1])
    m = y.size

    for i in range(iteration_number):
        output = calculate_output(X,theta)
        theta = theta - ((alpha / m) * (np.dot(X.T, (output - y))))
        print(calculate_cost(X, y, theta))

    return theta
    
def main():
    X, y = load_dataset()
    X = feature_standardization(X)
    theta = gradient_descent(X, y, 0.0001, 100000) 
    print("Theta's: {}".format(theta))

if __name__ == "__main__":
    main()

