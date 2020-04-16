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

print(lables)
# Encode labels, changing from words to numbers
lables = label_encoder.fit_transform(lables)
print(lables)

#Defining a trainingset
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(iris_data), lables, test_size=0.4, random_state=0)

print('Linear Regression')
#TRaining the model
model = linear_model.LinearRegression()
model.fit(trainData, trainLabels)


print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)

predictions = model.predict(testData)
df = pd.DataFrame({'Actual': testLabels.flatten(), 'Predicted': predictions.flatten()})
print(df)

#plt.scatter(trainData[:,0], trainLabels)
#plt.scatter(trainData[:,1], trainLabels)
#plt.scatter(trainData[:,2], trainLabels)
#plt.scatter(trainData[:,3], trainLabels)

plt.plot(testData, predictions, color='red', linewidth=2)

plt.scatter(testData[:,0], testLabels)
plt.scatter(testData[:,1], testLabels)
plt.scatter(testData[:,2], testLabels)
plt.scatter(testData[:,3], testLabels)
plt.show()

print('\n')
print('Linear discriminant classifier')
