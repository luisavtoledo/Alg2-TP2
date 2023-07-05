import pandas as pd
import numpy as np
import math
from k_centros import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time

#Tratamento de dados/manipulacao dos datasets para o uso no algoritmo

dataset1 = pd.read_csv("datasets/banknote.txt", header=None)
dataset1 = dataset1.astype(float)

dataset2 = pd.read_csv("datasets/HTRU_2.csv", header=None)
dataset2 = dataset2.astype(float)

dataset3 = pd.read_csv("datasets/magic04.csv", header=None)
dataset3 = dataset3.replace(['g'], 0)
dataset3 = dataset3.replace(['h'], 1)
dataset3 = dataset3.astype(float)

dataset4 = pd.read_csv("datasets/segmentation.csv", header=None)
dataset4 = dataset4.replace(['BRICKFACE'], 0)
dataset4 = dataset4.replace(['SKY'], 1)
dataset4 = dataset4.replace(['FOLIAGE'], 2)
dataset4 = dataset4.replace(['CEMENT'], 3)
dataset4 = dataset4.replace(['WINDOW'], 4)
dataset4 = dataset4.replace(['PATH'], 5)
dataset4 = dataset4.replace(['GRASS'], 6)
atr = dataset4.columns.tolist()
atr = atr[1:] + [atr[0]]
dataset4 = dataset4[atr]
dataset4 = dataset4.astype(float)

dataset5 = pd.read_csv("datasets/transfusion.csv", header=None)

dataset6 = pd.read_csv("datasets/trial.csv", header=None)

dataset7_1 = pd.read_csv("datasets/urbanGB.txt", header=None)
dataset7_2 = pd.read_csv("datasets/urbanGB.labels.txt", header=None)
dataset7 = pd.concat([dataset7_1, dataset7_2], axis=1, join='inner')

dataset8 = pd.read_csv("datasets/winequality-red.csv", sep=';', header=None)

dataset9 = pd.read_csv("datasets/winequality-white.csv", sep=';', header=None)

dataset10 = pd.read_csv("datasets/yeast.csv", sep='\s+', header=None)
dataset10 = dataset10.replace(['CYT'], 0)
dataset10 = dataset10.replace(['NUC'], 1)
dataset10 = dataset10.replace(['MIT'], 2)
dataset10 = dataset10.replace(['ME3'], 3)
dataset10 = dataset10.replace(['ME2'], 4)
dataset10 = dataset10.replace(['ME1'], 5)
dataset10 = dataset10.replace(['EXC'], 6)
dataset10 = dataset10.replace(['VAC'], 7)
dataset10 = dataset10.replace(['POX'], 8)
dataset10 = dataset10.replace(['ERL'], 9)
dataset10 = dataset10.drop(dataset10.columns[[0]], axis=1)

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
    #ponos = dataset.astype(float)
   # n_dataset = (dataset.iloc[:, 0:-1]).astype(float)
    #pontos = pd.read_csv(dataset)
    n_pontos = (dataset.iloc[:, 0:-1]).astype(float)
    dados = (n_pontos).to_numpy()
    classes = dataset.iloc[:, -1]
    k = classes.nunique()
    
    #Armazenando metricas
    tempo_kcentros = []
    raio_kcentros = []
    silhouette_kcentros = []
    rand_kcentros = []
    tempo_kmeans = []
    raio_kmeans = []
    silhouette_kmeans = []
    rand_kmeans = []

    matriz = matriz_dist(n_pontos, p)

    for i in range(30):
        inicio = time.time()
        labels, i_centros = k_centros(matriz, dataset, k, p)
        raio = maior_distancia(matriz, pd.DataFrame(i_centros))[1]
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

main(dataset4)