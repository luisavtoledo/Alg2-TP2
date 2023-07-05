import pandas as pd
import numpy as np
import math
from k_centros import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time

def dist_kmeans(ponto, centros, p):
    menor_dist = math.inf
    i_centro = 0
    t = centros.shape[0]

    for i  in range(t):
        dist = dist_Minkowski(ponto, centros[i], p)
        if dist < menor_dist:
            menor_dist = dist
            i_centro = i

    return (menor_dist, i_centro)

def raio_kmeans(pontos, centros, p):
    raio = 0
    t = pontos.shape[0]

    for i in range(t):
        dist = dist_kmeans(pontos[i], centros, p)[0]
        if dist > raio:
            raio = dist
    
    return raio


def testes(dataset, p):
    dados = dataset.iloc[:, 0:-1]
    classes = dataset.iloc[:, [-1]]
    k = classes.nunique()
    dados = dados.to_numpy

    #Armazenando metricas
    tempo_kcentros = []
    raio_kcentros = []
    silhouette_kcentros = []
    rand_kcentros = []
    tempo_kmeans = []
    raio_kmeans = []
    silhouette_kmeans = []
    rand_kmeans = []

    matriz_dist = matriz_dist(dados, p)

    for i in range(30):
        inicio = time.time()
        labels, i_centros = k_centros(matriz_dist, dados, k)
        raio = maior_distancia(matriz_dist, i_centros)[1]
        sil = silhouette_score(dados, labels)
        rand = adjusted_rand_score(classes, labels)
        raio_kcentros.append(raio)
        silhouette_kcentros.append(sil)
        rand_kcentros.append(rand)
        fim = time.time()
        tempo = fim - inicio
        tempo_kcentros.append(tempo)

        k_inicio = time.time()
        kmeans = KMeans(n_clusters=k).fit(dados)
        k_raio = raio_kmeans(dados, kmeans.cluster_centers_, p)
        k_sil = silhouette_score(dados, kmeans.labels_)
        k_rand = adjusted_rand_score(classes, kmeans.labels_)
        raio_kmeans.append(k_raio)
        silhouette_kmeans.append(k_sil)
        rand_kmeans.append(k_rand)
        k_fim = time.time()
        k_tempo = k_fim - k_inicio
        tempo_kmeans.append(k_tempo)

    print("kcentros:")
    print("Media tempo: ", np.mean(np.array(tempo_kcentros)))
    print("DP tempo: ", np.std(np.array(tempo_kcentros)))
    print("Media raio: ", np.mean(np.array(raio_kcentros)))
    print("DP raio: ", np.std(np.array(raio_kcentros)))
    print("Media silhouette: ", np.mean(np.array(silhouette_kcentros)))
    print("DP silhouette: ", np.std(np.array(silhouette_kcentros)))
    print("Media rand: ", np.mean(np.array(rand_kcentros)))
    print("DP rand: ", np.std(np.array(rand_kcentros)))

    print("kmeans:")
    print("Media tempo: ", np.mean(np.array(tempo_kmeans)))
    print("DP tempo: ", np.std(np.array(tempo_kmeans)))
    print("Media raio: ", np.mean(np.array(raio_kmeans)))
    print("DP raio: ", np.std(np.array(raio_kmeans)))
    print("Media silhouette: ", np.mean(np.array(silhouette_kmeans)))
    print("DP silhouette: ", np.std(np.array(silhouette_kmeans)))
    print("Media rand: ", np.mean(np.array(raio_kmeans)))
    print("DP rand: ", np.std(np.array(rand_kmeans)))
        
def main(dataset):
    print("p = 1")
    testes(dataset, 1)
    print("p = 2")
    testes(dataset, 2)