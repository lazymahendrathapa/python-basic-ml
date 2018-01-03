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

def calculate_cost(X,y, theta, lambda_):
    m = y.size
    output = calculate_output(X, theta)
    
    error = np.sum(np.square(output - y))
    regularize_term = lambda_ * np.sum(np.square(theta[1:]))

    return (error + regularize_term) / (2 * m)

def gradient_descent(X, y, alpha, iteration_number, lambda_):

    theta = np.zeros([X.shape[1],1])
    m = y.size

    for i in range(iteration_number):
        output = calculate_output(X,theta)
        sum_ = np.dot(X.T, (output - y))

        temp = np.array(theta)
        temp[0] = 0

        theta = theta - (alpha / m) * (sum_ + lambda_ * temp)

        if i % 1000 == 0:
            print("i: {}, Cost: {}".format(i, calculate_cost(X, y, theta, lambda_)))

    return theta
 
def main():
    X, y = load_dataset()
    X = feature_standardization(X)
    theta = gradient_descent(X, y, 0.001, 10000, 1.0) 
    print("Theta's: {}".format(theta))

if __name__ == "__main__":
    main()
