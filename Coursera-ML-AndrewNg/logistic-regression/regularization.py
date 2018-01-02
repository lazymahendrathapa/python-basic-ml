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

def calculate_cost(X, y, theta, lambda_):
    h = calculate_output(X, theta)
    general_cost = np.mean(-y * np.log(h) - (1 - y) * np.log(1-h)) 
    regularized_cost = (lambda_ / 2) * np.mean(np.power(theta[1:], 2))

    return general_cost + regularized_cost

def gradient_descent(X, y, alpha, iteration_number, lambda_):

    theta = np.zeros([X.shape[1], 1])
    m = y.size
    
    for i in range(iteration_number):
        output = calculate_output(X, theta)
        sum_ = np.dot(X.T, (output - y))
        temp = np.array(theta)
        temp[0] = 0
        
        theta = theta - (alpha/m) * (sum_ + lambda_ * temp)
        
        if i % 100 == 0:
            print("i: {}, Cost: {}".format(i, calculate_cost(X, y, theta, lambda_)))
    
    return theta

def main():
    X, y = load_dataset()
    theta = gradient_descent(X, y, 0.01, 1000, 1.0)
    print("Theta's: \n{}".format(theta))

if __name__ == "__main__":
    main()


