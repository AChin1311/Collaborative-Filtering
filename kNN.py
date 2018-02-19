import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.similarities import pearson
import matplotlib.pyplot as plt
from surprise.model_selection.split import KFold
from surprise import accuracy
import numpy as np

file_path = os.path.expanduser('ml-latest-small/ratings_new.csv')
reader = Reader(sep=',')
data = Dataset.load_from_file(file_path, reader=reader)
# data = Dataset.load_builtin('ml-100k')

sim_options = {'name': 'pearson',
              'user_based': False
              }

avg_rmse = []
avg_mae = []
all_k = []

kf = KFold(n_splits=10)
for i in range(2,102,10):
  print('k = ',i)
  all_k.append(i)
  knn = KNNWithMeans(k=i, sim_options=sim_options)
  # output = cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=10,  verbose=True)
  # avg_rmse.append(np.mean(output['test_rmse']))
  # avg_mae.append(np.mean(output['test_mae']))
  rmse=[]
  mae=[]
  for trainset, testset in kf.split(data):
    knn.fit(trainset)
    predict = knn.test(testset)
    rmse.append(accuracy.rmse(predict, verbose=True))
    mae.append(accuracy.mae(predict, verbose=True))
  avg_rmse.append(np.means(rmse))
  avg_mae.append(np.means(mae))

plt.plot(all_k,avg_rmse)
plt.savefig('plot/rmse_k.png')
plt.clf()

plt.plot(all_k,avg_mae)
plt.savefig('plot/mae_k.png')
plt.clf()


