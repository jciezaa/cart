import pandas
import graphviz
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import tree
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn import preprocessing

# Load Dataset
wine = 'winequality-white.csv'
data = pandas.read_csv(wine, sep=';', header=0)

# Grafico de dispersión
# scatter_matrix(data)
# plt.show()

# Split training dataset vs validation dataset
array = data.values
X = array[:,0:-1]
Y = array[:,-1]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# make predictions
cart = DecisionTreeRegressor()
cart.fit(X_train, Y_train)

predictions_1 = cart.predict(X_validation)
cart.score(X_validation, Y_validation)

# grafico del arbol
#feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
#dot_data = tree.export_graphviz(cart, out_file=None, max_depth=3, feature_names=feature_names, filled=True, rounded=True, special_characters=True)
#graph = graphviz.Source(dot_data)
#graph.render("wine")


# Plot the results
#plt.figure()
#plt.scatter(X_train[:,0], Y_train, s=20, edgecolor="black", c="darkorange", label="data")
#plt.plot(X_validation, predictions_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()