import numpy as np
import input_data
import DL

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    data = mnist.train.next_batch(8000)
    train_x = data[0]
    Y = data[1]
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    tb = mnist.train.next_batch(2000)
    Y_test = tb[1]
    X_test = tb[0]
    
    dims = [784, 100, 80, 50, 10]
    DL.model_DL(train_x.T, train_y.T, Y, X_test.T, Y_test, dims, alpha=0.5, num_iterations=1100, print_cost=True)

    return

main()