import pandas as pd
import numpy as np

#Calcula a distancia de Minkowski
def dist_Minkowski(X, Y, p):
    somatorio = np.sum(np.power(np.abs(X - Y), p))
    return np.power(somatorio, 1/p)

#Computa a matriz de distancia
def matriz_dist(S, p):
    t = S.shape[0]
    matriz = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            matriz[i][j] = dist_Minkowski(S[i], S[j], p)
    return matriz

#Acha o maior raio entre um ponto e um centro
def maior_raio(matriz_dist, i_centros):

    #return ponto

#Algoritmo 2-aproximado para o problema dos k-centros
def k_centros(S, k, p):

    #Computa a matriz de distancias para o dataset
    distancias = matriz_dist(S, p)

    #Se k for menor ou igual a quantidade de pontos, retorna o dataset
    t = S.shape[0] 
    if k >= t:
        return S
    else:
        #Cria o conjunto com indices da solucao e adiciona um ponto arbitrario
        i_centros = np.array([])
        primeiro_centro = np.random.randint(0, t)
        i_centros.append(primeiro_centro)

        #Adiciona recursivamente o ponto mais distante dos centros
        while len(i_centros) < k:
            novo_centro = maior_raio(distancias, i_centros)
            i_centros.append(novo_centro)

    #Cria o conjunto solucao 
    centros = np.array([])
    for i in range(len(i_centros)):
        indice = i_centros[i]
        centros.append(S[indice])

    return centros