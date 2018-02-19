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
              'user_based': True
              }
avg_rmse = []
avg_mae = []
all_k = []
prev_rmse = 2
prev_mae = 2

for i in range(2,102,2):
  print('k = ',i)
  all_k.append(i)
  knn = KNNWithMeans(k=i, sim_options=sim_options)
  output = cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=10,  verbose=True)
  
  avg_rmse.append(np.mean(output['test_rmse']))
  avg_mae.append(np.mean(output['test_mae']))

  print("average: ", avg_rmse[-1], avg_mae[-1])
  
  prev_rmse = np.mean(output['test_rmse'])
  prev_mae = np.mean(output['test_mae'])

min_k = 2*avg_rmse.index(min(avg_rmse))
k = min_k
while k > 0:
  if avg_rmse[k-1] - avg_rmse[k] > 0.002:
    break
  k -= 1
print(avg_rmse[:min_k])
print("min k: ", k)

plt.plot(all_k,avg_rmse)
plt.savefig('plot/rmse_k.png')
plt.clf()

plt.plot(all_k,avg_mae)
plt.savefig('plot/mae_k.png')
plt.clf()


