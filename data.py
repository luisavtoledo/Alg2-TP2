import pandas as pd

#Tratamento de dados/manipulacao dos datasets para o uso no algoritmo

dataset1 = pd.read_csv("datasets/banknote.txt", header=None)

dataset2 = pd.read_csv("datasets/HTRU_2.csv", header=None)

dataset3 = pd.read_csv("datasets/magic04.csv", header=None)
dataset3 = dataset3.replace(['g'], 0)
dataset3 = dataset3.replace(['h'], 1)

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

dataset5 = pd.read_csv("datasets/transfusion.csv", header=None)

dataset6 = pd.read_csv("datasets/trial.csv", header=None)

dataset7_1 = pd.read_csv("datasets/urbanGB.txt", header=None)
dataset7_2 = pd.read_csv("datasets/urbanGB.labels.txt", header=None)
dataset7 = pd.concat([dataset7_1, dataset7_2], axis=1, join='inner')
print(dataset7)
#pegar classes e mudar valores

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