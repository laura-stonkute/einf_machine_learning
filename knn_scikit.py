import pandas as pd
import numpy as np
import sklearn.neighbors as neigh
import sklearn.model_selection as modsel

column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

iris_data_set = pd.read_csv("iris.csv", names=column_names, header=None)  # csv file wird gelesen und die Daten werden nach den Kolonen-Namen sortiert


print(iris_data_set.head())  # Anfang vom Datenset
print(iris_data_set.tail())   # Schluss vom Datenset
print(iris_data_set.describe())  # Datenset wird beschrieben (zählen etc)

matrix_X = iris_data_set.iloc[0:150, 0:4].values  # features aller 150 samples werden in einer Matrix in 4er-Reihen widergegeben
vector_Y = iris_data_set.iloc[0:150, 4].values  # Klassen aller 150 samples werden in einem Vektor in 4er-Reihen widergegeben
#print(matrix_X)
#print(vector_Y)

X_train, X_test, Y_train, Y_test = modsel.train_test_split(matrix_X, vector_Y, test_size=0.2, random_state=0, stratify=vector_Y)  # split ist nicht regelmässig -> darum random_state um split zu fixen
                                                                                                                                # stratify=

classifier = neigh.KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)

# Teste die Samples mit dem Klassifizierer
# Vergleiche Vorhersage mit dem tatsächlichen Resultat
accuracy = classifier.score(X_test, Y_test)
print(f"Genauigkeit: {accuracy}")

error = 0
for index in range(len(Y_test)):
    actual_class = Y_test[index]
    sample = X_test[index, :]
    sample_row_vector = sample.reshape((1, 4))
    predicted_class = classifier.predict(sample_row_vector)[0]
    print(f"Vorhergesagte Klasse: {predicted_class}, Tatsächliche Klasse: {actual_class}")
    if actual_class != predicted_class:
        error += 1
print(error)
print(f"Genauigkeit (bei hand): {1-error/len(Y_test)}")

x = np.array([5.84, 3.05, 3.76, 1.20])
x_row_vector = x.reshape((1, 4))
predicted_class = classifier.predict(x_row_vector)
print(f"Klassifikation: {predicted_class}")







