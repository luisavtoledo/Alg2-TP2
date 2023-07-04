import pandas as pd
import numpy as np
import math
from k_centros import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time

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

    


def main(dataset):
    print("p = 1")
    testes(dataset, 1)
    print("p = 2")
    testes(dataset, 2)