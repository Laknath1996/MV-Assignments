import numpy as np
import pickle

from sklearn.metrics import accuracy_score


# define useful functions

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_from_file(file):
    dict = unpickle("/Users/ashwin/Semester 7/Machine Vision/Assignments/Assignment_2/datasets/cifar-10-batches-py/"+file)
    print("Unpacking {}".format(dict[b'batch_label']))
    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Y,names


def get_data():
    x_train = np.empty((3072,0)).astype("uint8")
    y_train = np.empty((10,0))
    n_train = np.empty(0)
    for b in range(1,6):
        fn = 'data_batch_' + str(b)
        X, Y, names = get_data_from_file(fn)
        x_train= np.append(x_train, X, axis=1)
        y_train= np.append(y_train, Y, axis=1)
        n_train= np.append(n_train, names)
    del X, Y
    fn = 'test_batch'
    x_test, y_test, n_test = get_data_from_file(fn)
    return x_train, y_train, n_train, x_test, y_test, n_test


def get_label_names(file):
    dict = unpickle("/Users/ashwin/Semester 7/Machine Vision/Assignments/Assignment_2/datasets/cifar-10-batches-py/"+file)
    L = np.asarray(dict[b'label_names'])
    return L

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoid_grad(z):
    return z*(1-z)

def softmax(z):
   # exps = np.exp(z - z.max())
    return np.exp(z) / sum(np.exp(z))

def relu(z):
    return np.maximum(0,z)

def relu_grad(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def layer_sizes(X, Y, n_h):
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(0)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1-Y),np.log(1-A2))
    cost = -(1/m)*np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']

    dZ2 = (A2 - Y)*sigmoid_grad(A2)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*relu_grad(A1)
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, lr, num_iterations = 1000, print_cost=False):
    np.random.seed(0)

    (n_x, n_h, n_y) = layer_sizes(X, Y, n_h)

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=lr)

        if print_cost :
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

def nn_model_mini_batching(X, Y, n_h, lr, batch_size=32, epochs=1000, decay=1e-6):
    np.random.seed(0)

    (n_x, n_h, n_y) = layer_sizes(X, Y, n_h)

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    m = X.shape[1]
    num_batches = int(np.floor(m/batch_size))

    # shuffle the training data
    idx = np.arange(m)
    np.random.shuffle(idx)
    x_train = X[:, idx]
    y_train = Y[:, idx]

    iter = 0
    for i in range(epochs):
        for j in range(num_batches+1):

            if j == num_batches :
                x_batch = x_train[:, j*batch_size:m]
                y_batch = y_train[:, j*batch_size:m]

            else:
                x_batch = x_train[:, j*batch_size:(j+1)*batch_size]
                y_batch = y_train[:, j*batch_size:(j+1)*batch_size]

            A2, cache = forward_propagation(x_batch, parameters)
            cost = compute_cost(A2, y_batch, parameters)
            lr = lr / (1 + decay*iter)
            iter += 1
            grads = backward_propagation(parameters, cache, x_batch, y_batch)
            parameters = update_parameters(parameters, grads, learning_rate=lr)

        print("Cost after epoch %i: %f" % (i, cost))

    return parameters

def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    # predictions = (A2 > 0.5)
    predictions = np.argmax(A2, axis=0)

    return predictions

##################################################################
# execution
##################################################################

print("Starting ...")

L = get_label_names('batches.meta')
x_train, y_train, n_train, x_test, y_test, n_test = get_data()

# select the dataset
labels = np.zeros((np.size(x_train, 1),))
for i in range(np.size(x_train, 1)):
    labels[i] = int(np.where(y_train[:, i]==1)[0])

X = np.empty((3072, 0)).astype("uint8")
y = np.empty((10, 0))
l = np.empty((0, ))
X_test = np.empty((3072, 0)).astype("uint8")
y_test = np.empty((10, 0))
l_test = np.empty((0, ))
for i in range(10):
    idx = np.where(labels == i)[0]
    X = np.append(X, x_train[:, idx[:100]], axis=1)
    y = np.append(y, y_train[:, idx[:100]], axis=1)
    l = np.append(l, labels[idx[:100]])
    X_test = np.append(X_test, x_train[:, idx[100:150]], axis=1)
    y_test = np.append(y_test, y_train[:, idx[100:150]], axis=1)
    l_test = np.append(l_test, labels[idx[100:150]])

# standardize
X = X.astype('float32')
X_test = X_test.astype('float32')
x_train = x_train.astype('float32')
x_train /= 255
X /= 255
X_test /= 255

print('X.shape = ', X.shape)
print('Y.shape = ', y.shape)
print('X_test.shape = ', X_test.shape)
print('Y_test.shape = ', y_test.shape)

# train
n_h = 512
parameters = nn_model_mini_batching(X, y, n_h, lr=0.01, batch_size=32, epochs=1000, decay=0)

# predict
pred = predict(parameters, X)
pred_test = predict(parameters, X_test)

print('Training Accuracy : %0.2f %%' % (accuracy_score(l, pred)*100))
print('Testing Accuracy : %0.2f %%' % (accuracy_score(l_test, pred_test)*100))

