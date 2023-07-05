import numpy as np
import math

#Calcula a distancia de Minkowski
def dist_Minkowski(X, Y, p):
    somatorio = np.sum(np.power(np.abs(np.array(X) - np.array(Y)), p))
    return np.power(somatorio, 1/p)

#Computa a matriz de distancia
def matriz_dist(S, p):
    t = S.shape[0]
    matriz = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            matriz[i][j] = dist_Minkowski(S.iloc[i], S.iloc[j], p)
    return matriz

#Acha o maior raio entre um ponto e um centro
def maior_distancia(matriz_dist, i_centros):
    ponto_distante = 0
    maior_raio = 0
    t = matriz_dist.shape[0]

    for i in range(t):
        distancia = math.inf
    
        #Encontra a distancia entre o ponto e o seu centro
        for j in range(len(i_centros)):
            if matriz_dist[i][i_centros[j]] < distancia:
                distancia = matriz_dist[i][i_centros[j]]

        if distancia > maior_raio:
            #Substitui o maior raio até agora pela distancia encontrada
            maior_raio = distancia
            ponto_distante = i

    return (ponto_distante, maior_raio)

#Algoritmo 2-aproximado para o problema dos k-centros
def k_centros(matriz_dist, S, k):
    
    #Se k for menor ou igual a quantidade de pontos, retorna o dataset
    t = S.shape[0] 
    if k >= t:
        return S
    else:
        #Cria o conjunto com indices da solucao e adiciona um ponto arbitrario
        i_centros = []
        primeiro_centro = np.random.randint(0, t)
        i_centros.append(primeiro_centro)

        #Adiciona recursivamente o ponto mais distante dos centros
        for i in range (k-1):
            novo_centro = maior_distancia(matriz_dist, i_centros)[0]
            i_centros.append(novo_centro)

    #Cria o conjunto solucao 
    centros = []
    for i in range(len(i_centros)):
        indice = i_centros[i]
        centros.append(S[indice])

    return (centros, i_centros)