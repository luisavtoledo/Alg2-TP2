import pandas as pd
import numpy as np

#Distancia de Minkowski
def dist_Minkowski(X, Y, p):
    somatorio =np.sum(np.power(np.abs(np.array(X) - np.array(Y)), p))
    return np.power(somatorio, 1/p)

#Matriz de distancia
def matriz_dist(X, p):
    t = X.shape[0]
    matriz = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            matriz[i][j] = dist_Minkowski(X[i], X[j], p)
    return matriz

#Maior raio

#Algoritmo 2-aproximado para o problema dos k-centros