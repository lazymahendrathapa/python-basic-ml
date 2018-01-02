import numpy as np

def load_dataset():
    dataset = np.genfromtxt('ex2data2.txt', delimiter=',')
    np.random.shuffle(dataset)
    X = dataset[:, :-1]
    y = dataset[:, [-1]]
    X = np.insert(X, 0, 1, axis = 1)
    
    return X, y

def calculate_output(X, theta):
    z = X.dot(theta)
    return 1 / (1 + np.exp(-z))

def calculate_cost(X, y, theta):
    h = calculate_output(X, theta)
    return np.mean(-y * np.log(h) - (1 - y) *  np.log(1 - h))

def gradient_descent(X, y, alpha, iteration_number):

    theta = np.random.random([X.shape[1], 1]) 
    m = y.size #number of training examples
    
    for i in range(iteration_number):
        output = calculate_output(X, theta)
        sum_ = np.dot(X.T, (output -y ))
        theta = theta - ((alpha / m ) * (sum_))

        if i % 100 == 0:
            print("i: {}, Cost: {}".format(i, calculate_cost(X, y, theta)))

    return theta

def main():
    X, y = load_dataset()
    theta = gradient_descent(X, y, 0.05, 1000)
    print("Theta's: \n{}".format(theta))

if __name__ == "__main__":
    main()
