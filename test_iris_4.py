# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
from pandas import read_csv                     #
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt                 #

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Egen kode
from sklearn import linear_model
# Import label encoder
from sklearn import preprocessing
import pandas as pd

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

data = "Iris_TTT4275/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(data, names=names)

# Shape
shape = dataset.shape
print("Shape:", shape)

# Head
print(dataset.head(20))

# Descriptions
print(dataset.describe())

# Class Distribution
print(dataset.groupby('class').size())

# partition the data into training and testing splits, using 30 (60%) of each
# of the data for training and the remaining 20 (40%) of each for testing
array = dataset.values
iris_data = array[:,0:4]
lables = array[:,4]

#print(lables)
# Encode labels, changing from words to numbers
lables = label_encoder.fit_transform(lables)
#print(lables)

lables = lables.astype(numpy.float32)        #lables = float(lables) #Changing lables to float numbers

#Defining a trainingset
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(iris_data), lables, test_size=0.4, random_state=0)
'''
#plotting training dataset
plt.scatter(trainData[:,0], trainLabels)
plt.scatter(trainData[:,1], trainLabels)
plt.scatter(trainData[:,2], trainLabels)
plt.scatter(trainData[:,3], trainLabels)
plt.show()
'''
#First try: Linear Classifier y_i = w*xi + w0
#Coefficient matrix
w = np.array([[1.0,1.0,1.0,1.0], [1.0,1.0,1.0,1.0], [1.0,1.0,1.0,1.0]], dtype='float') #changed to float
print('\n','w=','\n', w)

#Bias
w0 = np.array([[0],[0],[0]])
print('\n','w_0=','\n', w0)

#Learning rate LR=alpha
LR = 0.001
#Epochs = iterations
epochs = 100
#Error
error = []
'''
#Cost function MSE = Mean Square Error

#Running 100 times
for epoch in range(epochs):
    #initialize cost to 0
    epoch_cost, cost_w, cost_w0 = 0, 0, 0

    for i in range(len(trainData)):
        #make prediction
        y_i = w*trainData[i] #+ w0
        #adding MSE
        epoch_cost += 0.5*(np.square(y_i-lables[i]))       #0.5*((y_i-lables[i])^2)         #0.5*(np.transpose(y_i-lables[i])*(y_i-lables[i]))

        for j in range(len(trainData)):
            #Calculating the gradient
            gradient_MSE_w = ((y_i - lables[j])*y_i*(1-y_i))*trainData[j]

            #increase cost of coefficients
            cost_w += gradient_MSE_w
        w = w - LR*gradient_MSE_w
    error.append(epoch_cost)

print('w after training:', '\n', w)
'''

#second try:

#Cost function MSE = Mean Square Error

#Running 100 times
for epoch in range(epochs):
    #initialize cost to 0
    epoch_cost, cost_w, cost_w0 = 0, 0, 0

    for i in range(len(trainData)):
        #make prediction
        y_i = w*trainData[i] + w0

        D_w = (-2/(len(trainData)) * sum(trainData[i] * (lables[i] - y_i)))  # Derivative wrt w
        D_w0 = (-2/(len(trainData))) * sum(lables[i] - y_i)  # Derivative wrt w0

        w = w - LR*D_w
        w0 = w0 - LR*D_w0

print('w after training:', '\n', w)
print('\n')
print('w0 after training:', '\n', w0)

# Making predictions
Y_pred = [[]]
for i in range(len(testData)):
    #make prediction
    y_i = w*testData[i] + w0
    np.append(Y_pred, y_i)

plt.scatter(testData[:,0], testLabels)
plt.scatter(testData[:,1], testLabels)
plt.scatter(testData[:,2], testLabels)
plt.scatter(testData[:,3], testLabels)
'''
plt.plot(testData[:,0], Y_pred, color='red')
plt.plot(testData[:,1], Y_pred, color='red')
plt.plot(testData[:,2], Y_pred, color='red')
plt.plot(testData[:,3], Y_pred, color='red')

df = pd.DataFrame({'Actual': testLabels.flatten(), 'Predicted': Y_pred.flatten()})
print(df)
'''
plt.show()

print('test labels:', testLabels)
print('\n', 'predicted:', Y_pred)
