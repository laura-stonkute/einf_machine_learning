import sklearn.svm as svm
import pandas as pd

data_set = pd.read_csv("abstrakt.csv", names=None, header=None)
matrix_X = data_set.iloc[0:12, 0:2].values
vector_Y = data_set.iloc[0:12, 2].values

# Support Vector Classification
svclassifier = svm.SVC(kernel="linear")
svclassifier.fit(matrix_X, vector_Y)
print("w = ", svclassifier.coef_)
print("b = ", svclassifier.intercept_)
print("Number of supporting vectors:", svclassifier.n_support_)
print("Support Vectors", svclassifier.support_vectors_)

klasse_1 = svclassifier.predict([[8, 8]])
print(klasse_1)
klasse_2 = svclassifier.predict([[14, 14]])
print(klasse_2)

