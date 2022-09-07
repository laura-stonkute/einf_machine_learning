import numpy as np
from numpy import linalg as LA

# Vektor erstellen
vektor_1 = np.array([[1, 2, 3], [4, 5, 6]])
vektor_2 = np.array([4, 5, 6]).reshape((3, 1))
vektor_3 = np.array([1, 2, 3])
# linalg = lineare Algebra, norm => Betrag eines Vektors
norm = LA.norm(vektor_1)
print(norm)
print(LA.norm(vektor_1))
# dimension
print(vektor_1.ndim)
print(np.shape(vektor_1))
# Vektor spezifisch reshapen
print(vektor_1.reshape(6, 1))
# Transponieren
print(vektor_1.transpose())
# Skalarpeodukt -> Klammer für Matrix -> ein Vektor muss ein Zeilenvektor und der andere ein Spaltenvektor sein
print(np.dot(vektor_3, vektor_2))


