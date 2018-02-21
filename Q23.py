import csv
import os
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.dataset import DatasetAutoFolds
import matplotlib.pyplot as plt
import numpy as np

moiveId = []
moiveName = []
with open("ml-latest-small/ratings_new.csv", "r") as csvfile:
  content = csv.reader(csvfile)
  for row in content:
    uId, mId, name = row[0], row[1], row[3]
    if mId not in moiveId:
      moiveId.append(mId)
      moiveName.append(name)
csvfile.close()

dic = {}
with open("ml-latest-small/movies_new.csv", "r") as csvfile:
  content = csv.reader(csvfile)
  for row in content:
    mId, name, types = row[0], row[1], row[2].split('|')
    dic[mId] = types


file_path = os.path.expanduser('ml-latest-small/ratings_new.csv')
reader = Reader(sep=',')
data = DatasetAutoFolds(ratings_file=file_path, reader=reader).build_full_trainset()

nmf = NMF(n_factors=20)
nmf.fit(data)

for i in range(20):
  array = nmf.qi[:, i]
  top10 = np.argsort(array)[-10:]
  top10id = [moiveId[t] for t in top10]
  # print(top10id)
  print([moiveName[t] for t in top10])
  
  type_list = {}
  for mId in top10id:
    types = dic[mId]
    for t in types:
      if t not in type_list:
        type_list[t] = 1
      else:
        type_list[t] += 1
  print(type_list)