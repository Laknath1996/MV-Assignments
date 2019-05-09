import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import preprocessing
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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


def compute_sift(im, sift):
    im = im.reshape(3, 32, 32).transpose([1, 2, 0])
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(im, None)
    return des


def get_hist(des, kmeans, no_clusters, flag):
    pred = kmeans.predict(des)
    hist, bin = np.histogram(pred, bins=range(no_clusters+1))
    if flag:
        plt.bar(bin, hist, align='center', alpha=0.5)
        plt.show()
    return hist.reshape((1, no_clusters))


def build_descriptor_list(X, sift):
    descriptor_list = np.empty((0, 128))
    for i in range(np.size(X, 0)):
        im = X[i, :]
        des = compute_sift(im, sift)
        if des is None:
            continue
        descriptor_list = np.append(descriptor_list, des, axis=0)
    return descriptor_list


def build_histogram(X, no_clusters, kmeans):
    histograms = np.empty((0, no_clusters))
    for i in range(np.size(X, 0)):
        im = X[i, :]
        des = compute_sift(im, sift)
        if des is not None:
            histograms = np.append(histograms, get_hist(des, kmeans, no_clusters, False), axis=0)
        else:
            histograms = np.append(histograms, np.zeros((1, no_clusters)), axis=0)
    return histograms


print("Starting ...")

L = get_label_names('batches.meta')
x_train, y_train, n_train, x_test, y_test, n_test = get_data()
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('n_train.shape = ', n_train.shape)
print('x_test.shape = ', x_test.shape)

visualize_image(x_train, y_train, n_train, L, 0)

# select 1000 images for training and 500 images for testing (equal number of images for each class)
labels = np.zeros((np.size(x_train, 1),))
for i in range(np.size(x_train, 1)):
    labels[i] = int(np.where(y_train[:, i]==1)[0])

X = np.empty((3072, 0)).astype("uint8")
Y = np.empty((0, ))
X_test = np.empty((3072, 0)).astype("uint8")
Y_test = np.empty((0, ))
for i in range(10):
    idx = np.where(labels == i)[0]
    X = np.append(X, x_train[:, idx[:100]], axis=1)
    Y = np.append(Y, labels[idx[:100]])
    X_test = np.append(X_test, x_train[:, idx[100:150]], axis=1)
    Y_test = np.append(Y_test, labels[idx[100:150]])

X = X.T
X_test = X_test.T

print('X.shape = ', X.shape)
print('Y.shape = ', Y.shape)
print('X_test.shape = ', X_test.shape)
print('Y_test.shape = ', Y_test.shape)


# build descriptor list
sift = cv.xfeatures2d.SIFT_create()
descriptor_list = build_descriptor_list(X, sift)

print('discriptor-list shape = ', descriptor_list.shape)

# making the vocabulary
no_clusters = 50
kmeans = KMeans(n_clusters=no_clusters)
kmeans.fit(descriptor_list)

# build the histograms
histograms = build_histogram(X, no_clusters, kmeans)
histograms_test = build_histogram(X_test, no_clusters, kmeans)
print('histograms shape = ', histograms.shape)
print('histograms_test shape = ', histograms_test.shape)

# standardize the data
sc = preprocessing.StandardScaler()
histograms = sc.fit_transform(histograms)
histograms_test = sc.transform(histograms_test)

# svm
clf = svm.LinearSVC(C=10)
clf.fit(histograms, Y)

# predict
Y_pred = clf.predict(histograms)
Y_test_pred = clf.predict(histograms_test)

# accuracies
print("Training Accuracy = %0.2f %%" % (accuracy_score(Y, Y_pred)*100))
print("Testing Accuracy = %0.2f %%" % (accuracy_score(Y_test, Y_test_pred)*100))

print('Done.')
