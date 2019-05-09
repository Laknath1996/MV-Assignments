print(__doc__)
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers
import math
from sklearn.metrics.pairwise import rbf_kernel


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, gamma=0.1):
    return np.exp(-gamma*np.linalg.norm(x-y)**2)

def binary_svm(X, y, kernel, C):
    N = X.shape[0]
    q = matrix(-np.ones((N, 1)), tc='d')
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j])
    P = matrix(np.outer(y, y) * K, tc='d')
    g1 = np.asarray(np.diag(np.ones(N) * -1))
    g2 = np.asarray(np.diag(np.ones(N)))
    G = matrix(np.append(g1, g2, axis=0), tc='d')
    h = matrix(np.append(np.zeros(N), (np.ones(N) * C), axis =0), tc='d')
    A = matrix(y.reshape(1, N), tc='d')
    b = matrix(np.zeros((1, 1)), tc='d')

    sol = solvers.qp(P, q, G, h, A, b)

    alpha = np.ravel(sol['x'])
    idx = alpha > 0
    ind = np.arange(len(alpha))[idx]
    alpha = alpha[idx]
    sv = X[idx]
    sv_y = y[idx]
    print("%d support vectors out of %d points" % (len(alpha), X.shape[0]))

    b = 0
    for i in range(len(alpha)):
        b += sv_y[i]
        b -= np.sum(alpha * sv_y * K[ind[i], idx])
    b /= len(alpha)

    clf = {}
    clf['alpha'] = alpha
    clf['sv'] = sv
    clf['sv_y'] = sv_y
    clf['b'] = b
    clf['kernel'] = kernel

    return clf

def compute_score(X, clf):
    alpha = clf['alpha']
    sv = clf['sv']
    sv_y = clf['sv_y']
    b = clf['b']
    kernel = clf['kernel']

    if kernel == linear_kernel:
        w = np.zeros(len(X))
        for n in range(len(alpha)):
            w += alpha[n]*sv_y[n]*sv[n]
        score = np.dot(X, w) + b
    else:
        score = 0
        for a_i, sv_y_i, sv_i in zip(alpha, sv_y, sv):
            score += a_i * sv_y_i * kernel(X, sv_i)
        score += b

    return score

def multi_svm(X, y, kernel, C):
    no_classes = len(np.unique(y))
    clf = {}
    for i in range(no_classes):
        yn = np.copy(y)
        yn[yn != i] = -1
        yn[yn == i] = 1
        clf['c%s' % (str(i))] = binary_svm(X, yn, kernel, C)
    return clf

def predict(X, clf):
    no_classes = len(clf)
    pred = []
    for i in range(X.shape[0]):
        scores = []
        for j in range(no_classes):
            scores.append(compute_score(X[i, :], clf['c%s' % (str(j))]))
        pred.append(np.argmax(scores))
    return pred

# import some data to play with
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# Split the dataset to train and test splits ( 70 % train and 30 % test )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

clf1 = multi_svm(X_train, y_train, gaussian_kernel, 100)
pred1_train = predict(X_train, clf1)
pred1_test = predict(X_test, clf1)
print(pred1_test)

clf2 = svm.SVC(C=100, gamma=0.1)
clf2.fit(X_train, y_train)
pred2_train = clf2.predict(X_train)
pred2_test = clf2.predict(X_test)
print(pred2_test)

print("Training Accuracy from SVM1 : ", accuracy_score(y_train, pred1_train))
print("Testing Accuracy from SVM1 : ", accuracy_score(y_test, pred1_test))
print("Training Accuracy from SVM2 : ", accuracy_score(y_train, pred2_train))
print("Testing Accuracy from SVM2 : ", accuracy_score(y_test, pred2_test))

fig, sub = plt.subplots(1, 2)
axes = sub.flatten()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
Z1 = predict(np.c_[xx.ravel(), yy.ravel()], clf1)
Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = np.array(Z1)
Z1 = Z1.reshape(xx.shape)
Z2 = Z2.reshape(xx.shape)
ax1 = axes[0]
ax1.contourf(xx, yy, Z1)
ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel('Sepal length')
ax1.set_ylabel('Sepal width')
ax1.set_xticks(())
ax1.set_yticks(())
ax2 = axes[1]
ax2.contourf(xx, yy, Z2)
ax2.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('Sepal length')
ax2.set_ylabel('Sepal width')
ax2.set_xticks(())
ax2.set_yticks(())

plt.show()







