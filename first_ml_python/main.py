# First Machine Learning Data Set

# Check the versions of libraries

# Python version
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

# Load libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print('Python: {}'.format(sys.version))
# scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Gets the Dimension of the dataset
print(dataset.shape)
# Gets the top of the dataset
print(dataset.head(20))
# Gets the Summary Statistics of the Dataset
print(dataset.describe())
# Gets the Class Distribution of the dataset
print(dataset.groupby('class').size())

# Since the data is univariate, plot box and whisker plots of each variable
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Print Histograms of the dataset

dataset.hist()
plt.show()

# Show the scatter plots of all variables
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# The plots tell us that there are some relationships between models, so we
# expect a certain level of Linearity

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions on validation dataset
svm = SVC()
svm.fit(X_train, Y_train)
svm_predictions = svm.predict(X_validation)
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
knn_predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, svm_predictions))
print(confusion_matrix(Y_validation, svm_predictions))
print(classification_report(Y_validation, svm_predictions))

print(accuracy_score(Y_validation, knn_predictions))
print(confusion_matrix(Y_validation, knn_predictions))
print(classification_report(Y_validation, knn_predictions))
