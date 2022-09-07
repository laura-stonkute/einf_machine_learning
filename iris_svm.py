import sklearn.svm as svm
import pandas as pd

column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
iris_data_set = pd.read_csv("iris.csv", names=column_names, header=None)

matrix_X = iris_data_set.iloc[0:150, 0:4].values
vector_Y = iris_data_set.iloc[0:150, 4].values

svclassifier = svm.SVC(kernel="linear")
svclassifier.fit(matrix_X, vector_Y)

