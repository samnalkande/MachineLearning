# Loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading datasets
iris = datasets.load_iris()

# Printing description and features
print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

# preds = clf.predict([[1, 1, 1, 1]]) # - Iris-Setosa
# preds = clf.predict([[31, 1, 1, 1]])  # - Iris-Versicolour
preds = clf.predict([[31, 12, 16, 18]])   # - Iris-Versicolour
print(preds)