import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.similarities import pearson
import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.expanduser('ml-latest-small/ratings_popular.csv')
reader = Reader(sep=',')
data = Dataset.load_from_file(file_path, reader=reader)
# data = Dataset.load_builtin('ml-100k')

sim_options = {'name': 'pearson',
              'user_based': True
              }

avg_rmse = []
avg_mae = []
all_k = []

for i in range(2,102,2):
  print('k = ',i)
  all_k.append(i)

  algo = KNNWithMeans(k=i, sim_options=sim_options)
  output = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10,  verbose=True, n_jobs=1)
  avg_rmse.append(np.mean(output['test_rmse']))
  avg_mae.append(np.mean(output['test_mae']))

print("min rmse k:", avg_rmse.index(min(avg_rmse)))
print("min mae k:", avg_mae.index(min(avg_mae)))

plt.plot(all_k,avg_rmse)
plt.savefig('plot/rmse_k_pop.png')
plt.clf()

plt.plot(all_k,avg_mae)
plt.savefig('plot/mae_k_pop.png')
plt.clf()

