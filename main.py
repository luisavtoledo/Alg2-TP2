import pandas as pd
import numpy as np
import math
from k_centros import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time

def testes(dataset, p):
    

def main(dataset):
    print("p = 1")
    testes(dataset, 1)
    print("p=2")
    testes(dataset, 2)