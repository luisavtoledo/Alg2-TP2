import pandas as pd
import numpy as np

#Distancia de Minkowski
def dist_Minkowski(X, Y, p):
    somatorio =np.sum(np.power(np.abs(np.array(X) - np.array(Y)), p))
    return np.power(somatorio, 1/p)

#Matriz de distancia

#Algoritmo 2-aproximado para o problema dos k-centros