import numpy as np
import scipy.io as sio
import scipy.optimize as optimize

def load_dataset():
    dataset = sio.loadmat('ex3data1.mat')
    X = dataset['X']
    y = dataset['y']
    X = np.insert(X, 0, 1, axis = 1) 
    
    return X, y

def calculate_output(X, theta):
    z = X.dot(theta)
    return 1 / (1 + np.exp(-z))

# theta must be the 1D array that are to be changed in the search for a minimum and args are
# the other(fixed) parameters.

def calculate_cost(theta, *args):

    X, y, lambda_ = args
    theta_temp = theta.reshape(theta.shape[0], -1)

    h = calculate_output(X, theta_temp)
    general_cost = np.mean(-y * np.log(h) - (1 - y) * np.log(1-h)) 
    regularized_cost = (lambda_ / 2) * np.mean(np.power(theta_temp[1:], 2))

    return general_cost + regularized_cost

def calcualte_gradient(theta, *args):

    theta_temp = theta.reshape(theta.shape[0], -1)

    X, y, lambda_ = args
    m = y.size
    output = calculate_output(X, theta_temp)

    temp = np.array(theta_temp)
    temp[0] = 0
    
    
    gradient =  (1/m) * (np.dot(X.T, (output - y)) + lambda_ * temp)

    return gradient.reshape(-1,)

def test(X, y, thetas):
    
    thetas = np.array(thetas)
    h = calculate_output(X, thetas.T)
    h = np.argmax(h, axis=1)
    h = h + 1
    y = y.reshape(-1, )
    
    acc = (h == y).sum()/h.shape
    print("Accuracy: {}".format(acc))

def train(X, y, num_label, iteration_number, lambda_):
     
    thetas = []

    for c in range(1, num_label + 1):

        initial_theta = np.zeros([X.shape[1],])
        args = (X, 1 * (y == c), lambda_)

        theta = optimize.fmin_cg(calculate_cost, initial_theta, fprime=calcualte_gradient, args = args)
        
        thetas.append(theta)
    
    return thetas

def main():
    X, y = load_dataset()
    thetas = train(X, y, 10, 1000, 2.0)
    test(X, y, thetas)

if __name__ == "__main__":
    main()

