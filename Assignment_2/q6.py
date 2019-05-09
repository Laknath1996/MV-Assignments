import numpy as np
from scipy.optimize import fmin_cg
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import log_loss
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


def visualize_image(X, Y, names, label_names, id):
    rgb = X[:,id]
    #print(rgb.shape)
    img = rgb.reshape(3,32,32).transpose([1, 2, 0]) #print(img.shape)
    plt.imshow(img)
    plt.title("%s%s%s" % (names[id], ', Class = ', label_names[np.where(Y[:,id]==1.0)]) )
    plt.show()
    #dir = os.path.abspath("output/samples")
    # #plt.savefig(dir+"/"+names[id].decode('ascii'))

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def relu(z):
    return z*(z > 0)

def sigmoidGrad(z):
    return sigmoid(z)*-sigmoid(z)

def initializeWeights(Lin, Lout):
    W = np.zeros((Lout, Lin+1))
    epsilon_init = 0.12
    W = np.random.rand(Lout, Lin+1)*2*epsilon_init-epsilon_init
    return W

def cost_function(Theta, input_layer_size, hidden_layer_size, num_labels, X, y):
    # reshape the Theta to weight matrices
    Theta1 = np.reshape(Theta[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))

    m = X.shape[0] # number of training samples

    # initialize
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    X = np.append(np.ones((m, 1)), X, axis=1)
    totalCost = 0

    for i in range(m):
        a1 = X[i, :]
        z2 = np.matmul(Theta1, a1.T)
        g1 = sigmoid(z2)
        a2 = g1.T
        a2 = np.append(1, a2)
        z3 = np.matmul(Theta2, a2.T)
        h = sigmoid(z3)

        y_i = y[i, :]

        # cost = np.sum(-y_i * np.log(h) - (-y_i * np.log(-h)))
        cost = log_loss(y_i, h)

        totalCost += cost

    J = totalCost/m

    delta_1 = np.zeros(Theta1.shape)
    delta_2 = np.zeros(Theta2.shape)

    for i in range(m):
        a1 = X[i, :]
        z2 = np.matmul(Theta1, a1.T)
        g1 = sigmoid(z2)
        a2 = g1.T
        a2 = np.append(1, a2)
        z3 = np.matmul(Theta2, a2.T)
        h = sigmoid(z3)

        y_i = y[i, :]

        d3 = h - y_i.T
        d2 = np.matmul(Theta2.T, d3) * sigmoidGrad(np.append(1, z2))

        delta_1 += np.matmul(d2[1:].reshape(len(d2[1:]),1), a1.reshape(len(a1),1).T)
        delta_2 += np.matmul(d3.reshape(len(d3), 1),a2.reshape(len(a2),1).T)

    Theta1_grad = delta_1 / m
    Theta2_grad = delta_2 / m

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten(), axis=0)

    return J, grad

def gradient_descent(Theta, alpha, num_iters, args):
    input_layer_size, hidden_layer_size, num_labels, X, y = args
    for i in range(num_iters):
        J, grad = cost_function(Theta, input_layer_size, hidden_layer_size, num_labels, X, y)
        print("iteration No. %d cost=%0.2f" % (i+1, J))
        Theta = Theta - alpha*grad
    return Theta


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    h1 = sigmoid(np.matmul(np.append(np.ones((m, 1)), X, axis=1), Theta1.T))
    h2 = sigmoid(np.matmul(np.append(np.ones((m, 1)), h1, axis=1), Theta2.T))

    pred = np.argmax(h2, axis=1)
    return pred

print("Starting ...")

L = get_label_names('batches.meta')
x_train, y_train, n_train, x_test, y_test, n_test = get_data()
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('n_train.shape = ', n_train.shape)
print('x_test.shape = ', x_test.shape)

# select 1000 images for training and 500 images for testing (equal number of images for each class)
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

X = X.T
X_test = X_test.T
y = y.T
y_test = y_test.T

# standardize
X = X.astype('float32')
X_test = X_test.astype('float32')
X /= 255
X_test /= 255

print('X.shape = ', X.shape)
print('Y.shape = ', y.shape)
print('X_test.shape = ', X_test.shape)
print('Y_test.shape = ', y_test.shape)

# set up the parameters
input_layer_size = 32*32*3
hidden_layer_size = 512
num_labels = 10

# initialize the parameters
initial_Theta1 = initializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = initializeWeights(hidden_layer_size, num_labels)

initial_Theta = np.append(initial_Theta1.flatten(), initial_Theta2.flatten(), axis=0)

# define the arguments
args = (input_layer_size, hidden_layer_size, num_labels, X, y)

# train
alpha = 0.001
num_iters = 100
Theta = gradient_descent(initial_Theta, alpha, num_iters, args)

Theta1 = np.reshape(Theta[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
Theta2 = np.reshape(Theta[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))

# predict
y_pred = predict(Theta1, Theta2, X)
y_test_pred = predict(Theta1, Theta2, X_test)

print("Training Accuracy : %0.2f %%" % (accuracy_score(y_pred, l) * 100))
print("Test Accuracy : %0.2f %%" % (accuracy_score(y_test_pred, l_test) * 100))





