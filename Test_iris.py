
# Tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

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
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

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

"""
# Box and Whiskers plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Histogram
dataset.hist()
pyplot.show()

# Scatter Plot Matrix
scatter_matrix(dataset)
pyplot.show()
"""
# Split-out Validation Setting
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2)

# Spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare algorithms
'''
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
'''

# Make predictions on validation dataset
print('\n LR')
model_1 = LogisticRegression(solver='liblinear', multi_class='ovr')
model_1.fit(X_train, Y_train)
predictions_1 = model_1.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions_1))
print(confusion_matrix(Y_validation, predictions_1))
print(classification_report(Y_validation, predictions_1))

print('\n LDA')
model_2 = LinearDiscriminantAnalysis()
model_2.fit(X_train, Y_train)
predictions_2 = model_2.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions_2))
print(confusion_matrix(Y_validation, predictions_2))
print(classification_report(Y_validation, predictions_2))
