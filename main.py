import pandas as pd
import numpy as np
import math
from k_centros import matriz_dist, k_centros, dist_Minkowski, maior_distancia
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time

#Tratamento de dados/manipulacao dos datasets para o uso no algoritmo

dataset1 = pd.read_csv("datasets/banknote.txt", header=None)
dataset1 = dataset1.astype(float)

dataset2 = pd.read_csv("datasets/tic-tac-toe.csv", header=None)
dataset2 = dataset2.replace(['x'], 0)
dataset2 = dataset2.replace(['o'], 1)
dataset2 = dataset2.replace(['b'], 2)
dataset2 = dataset2.replace(['positive'], 1)
dataset2 = dataset2.replace(['negative'], 0)

dataset3 = pd.read_csv("datasets/breast-cancer.csv", header=None)
dataset3 = dataset3.drop(dataset3.columns[[0]], axis=1)
dataset3 = dataset3.replace('?', np.nan)
dataset3 = dataset3.apply(pd.to_numeric, errors='coerce')
media = dataset3.mean()
dataset3 = dataset3.fillna(media)
dataset3.iloc[:, -1] = dataset3.iloc[:, -1].replace(2, 0)
dataset3.iloc[:, -1] = dataset3.iloc[:, -1].replace(4, 1)

dataset4 = pd.read_csv("datasets/cmc.csv", header=None)
dataset4.iloc[:, -1] = dataset4.iloc[:, -1].replace(1, 0)
dataset4.iloc[:, -1] = dataset4.iloc[:, -1].replace(2, 1)
dataset4.iloc[:, -1] = dataset4.iloc[:, -1].replace(3, 2)

dataset5 = pd.read_csv("datasets/transfusion.csv", header=None)

dataset6 = pd.read_csv("datasets/trial.csv", header=None)
dataset6 = dataset6.replace('SAFIDON', np.nan)
dataset6 = dataset6.replace('NUH', np.nan)
dataset6 = dataset6.replace('LOHARU', np.nan)
dataset6 = dataset6.apply(pd.to_numeric, errors='coerce')
media = dataset6.mean()
dataset6 = dataset6.fillna(media)

dataset7 = pd.read_csv("datasets/german.csv",sep='\s+',  header=None)
dataset7.iloc[:, -1] = dataset7.iloc[:, -1].replace(1, 0)
dataset7.iloc[:, -1] = dataset7.iloc[:, -1].replace(2, 1)

dataset8 = pd.read_csv("datasets/winequality-red.csv", sep=';', header=None)

dataset9 = pd.read_csv("datasets/mammographic_masses.csv", header=None)
dataset9 = dataset9.replace('?', np.nan)
dataset9 = dataset9.apply(pd.to_numeric, errors='coerce')
media = dataset9.mean()
dataset9 = dataset9.fillna(media)

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

#Calcula a disancia entre um ponto e seu centro
def dist_centro(ponto, centros, p):
    menor_dist = math.inf
    i_centro = 0
    t = centros.shape[0]

    for i  in range(t):
        dist = dist_Minkowski(ponto, centros[i], p)
        if dist < menor_dist:
            menor_dist = dist
            i_centro = i

    return (menor_dist, i_centro)

#Computa o raio final do algoritmo e a classificacao
def metricas(pontos, centros, p):
    raio = 0
    t = pontos.shape[0]
    labels = []

    for i in range(t):
        dist, centro = dist_centro(pontos[i], centros, p)
        labels.append(centro)
        if dist > raio:
            raio = dist
    
    return (raio, labels)

#Funcao de testes
def testes(dataset, p):
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
        centros, i_centros = k_centros(matriz, dados, k)
        raio = maior_distancia(matriz, i_centros)[1]
        labels = metricas(dados, centros, p)[1]
        sil = silhouette_score(dados, labels)
        rand = adjusted_rand_score(classes, labels)
        raio_kcentros.append(raio)
        silhouette_kcentros.append(sil)
        rand_kcentros.append(rand)
        fim = time.time()
        tempo = fim - inicio
        tempo_kcentros.append(tempo)

        k_inicio = time.time()
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(dados)
        k_raio = metricas(dados, kmeans.cluster_centers_, p)[0]
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
        
def main(datasets):
    t = len(datasets)
    for i in range(t):
        print("Dataset ", i+1)
        print("p = 1")
        testes(datasets[i], 1)
        print("p = 2")
        testes(datasets[i], 2)

datasets = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10]

main(datasets)