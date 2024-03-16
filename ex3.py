import numpy as np
import sys

# load the input of examples and test x set
arg_train_x, arg_train_y, arg_test_x = sys.argv[1], sys.argv[2], sys.argv[3]
train_x = np.loadtxt(arg_train_x) / 255.
train_y = np.loadtxt(arg_train_y).astype(int)
test_x = np.loadtxt(arg_test_x) / 255.
out_fname_file = open("test_y", 'w')


sigmoid = lambda x: 1 / (1 + np.exp(-x))

# shuffle the given examples while maintaining the correct classifications
def shuffle_set(x_arr, y_arr):
    # shuffle the given examples
    shuffled_examples = list(zip(x_arr, y_arr))
    np.random.shuffle(shuffled_examples)
    x_arr = np.array(list(zip(*shuffled_examples))[0])
    y_arr_shuff = np.array(list(zip(*shuffled_examples))[1])
    return x_arr, y_arr_shuff

# calculate softmax for z after substract max(z) from z
def softmax(z):
    e_x = np.exp(z - np.max(z))
    return e_x / e_x.sum()



def bprop(fprop_cach):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cach[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    h2 = h2.reshape((10, 1))
    h2[y] = h2[y] - 1
    dz2 = h2                                 # dL/dz2
    dW2 = np.dot(dz2, h1[np.newaxis, ...])   # dL/dz2 * dz2/dw2
    db2 = dz2                                # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cach['W2'].T,
       dz2) * sigmoid(z1/len(z1))[..., np.newaxis] * (1-(sigmoid(z1/len(z1)))[..., np.newaxis])  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x[..., np.newaxis].T)                        # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                                                      # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

def fprop(x, params_cache):
    # Follows procedure given in notes
    W_1, b_1, W_2, b_2 = [params_cache[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W_1, np.transpose(x)) + b_1.flatten()   # output z1 after moving to the hidden layer
    h1 = sigmoid(z1/len(z1))                            # sigmoid of z1
    z2 = np.dot(W_2, np.transpose(h1)) + b_2.flatten()  # output z2 after moving to the output layer
    h2 = softmax(z2)                                    # softmax of z2
    loss = sum([-np.log(i) for i in h2])                # calculate loss
    ret = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params_cache:
        ret[key] = params_cache[key]
    return ret

# predict the y of each x with given x set and params
def predict(test_x, params):
    y_hat_vec = []
    # the argmax of h2 is the appropriate class for x
    for i in range(len(test_x)):
        values_fprop_xi = fprop(test_x[i], params)
        y_hat_vec.append(np.argmax(values_fprop_xi['h2']))
    return y_hat_vec

def train_and_predict(train_set_x, train_set_y, validation_set_x, validation_set_y, test_x, params_cache, epochs, eta):
    # initialization for the best params
    max_success_rate_params = 0
    max_params_cache = params_cache

    for ep in range(epochs):
        shuffle_train_x, shuffle_train_y = shuffle_set(train_set_x, train_set_y)
        for i in range(len(train_set_x)):
            # frontpropagation
            values_fprop_xi = fprop(shuffle_train_x[i], params_cache)
            values_fprop_xi['y'] = shuffle_train_y[i]
            # backpropagation
            values_bprop_xi = bprop(values_fprop_xi)
            # update params
            params_cache['W1'] = params_cache['W1'] - eta * values_bprop_xi['W1']
            params_cache['b1'] = params_cache['b1'] - eta * values_bprop_xi['b1']
            params_cache['W2'] = params_cache['W2'] - eta * values_bprop_xi['W2']
            params_cache['b2'] = params_cache['b2'] - eta * values_bprop_xi['b2']

        # send the updated params to validation and store params if they have highest success rate
        valid_yhats=predict(validation_set_x, params_cache)
        success_rate_params = (len([i for i, j in zip(validation_set_y, valid_yhats) if i == j])/len(validation_set_x))*100
        if (success_rate_params>max_success_rate_params):
            max_params_cache=params_cache
            max_success_rate_params = success_rate_params
    # predict and return array of y_hat results
    return predict(test_x, max_params_cache)


if __name__ == '__main__':

    # separate to train and validation set
    len_train_set = int(0.95 * len(train_x))
    train_set_x = train_x[:len_train_set]
    validation_set_x = train_x[len_train_set:len(train_x)]
    train_set_y = train_y[:len_train_set]
    validation_set_y = train_y[len_train_set:len(train_x)]

    # initialize hyper-params
    epochs = 30
    eta = 0.1
    hidden_layer_dim = 50

    # initialize random parameters
    W1 = np.random.randn(hidden_layer_dim, 784) / np.sqrt(784)
    b1 = np.random.randn(hidden_layer_dim, 1) / np.sqrt(hidden_layer_dim)
    W2 = np.random.randn(10, hidden_layer_dim) / np.sqrt(hidden_layer_dim)
    b2 = np.random.randn(10, 1)/np.sqrt(10)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # train multi-class neural network algorithm and predict results with single hidden layer
    y_hat_arr = train_and_predict(train_set_x, train_set_y, validation_set_x, validation_set_y, test_x, params, epochs, eta)

    # write to output file
    for i in range(len(test_x)):
        out_fname_file.write(f"{y_hat_arr[i]}\n")



