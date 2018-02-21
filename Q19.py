import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.similarities import pearson
from surprise.prediction_algorithms.matrix_factorization import NMF
import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.expanduser('ml-latest-small/ratings_popular.csv')
reader = Reader(sep=',')
data = Dataset.load_from_file(file_path, reader=reader)
sim_options = {'name': 'pearson',
              'user_based': True
              }              


avg_rmse = []
avg_mae = []
all_k = []

for i in range(2,52,2):
  print('k = ',i)
  all_k.append(i)
  nmf = NMF(n_factors=i)
  output = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=10,  verbose=True)
  
  avg_rmse.append(np.mean(output['test_rmse']))
  avg_mae.append(np.mean(output['test_mae']))

print("min rmse k:", avg_rmse.index(min(avg_rmse)))
print("min mae k:", avg_mae.index(min(avg_mae)))

plt.plot(all_k,avg_rmse)
plt.savefig('plot/nmf_rmse_k_pop.png')
plt.clf()

plt.plot(all_k,avg_mae)
plt.savefig('plot/nmf_mae_k_pop.png')
plt.clf()


