import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.similarities import pearson
import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.expanduser('ml-latest-small/ratings_var.csv')
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
plt.savefig('plot/rmse_k_var.png')
plt.clf()

plt.plot(all_k,avg_mae)
plt.savefig('plot/mae_k_var.png')
plt.clf()




'''
                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Fold 6  Fold 7  Fold 8  Fold 9  Fold 10 Mean    Std     
RMSE (testset)    0.9785  0.9868  0.9842  0.9786  0.9815  0.9788  0.9794  0.9785  0.9813  0.9990  0.9827  0.0061  
MAE (testset)     0.7717  0.7763  0.7738  0.7702  0.7757  0.7722  0.7659  0.7701  0.7694  0.7806  0.7726  0.0040  
Fit time          3.42    3.63    3.68    3.52    3.60    3.56    3.48    3.36    2.35    2.12    3.27    0.53    
Test time         3.92    3.99    4.11    4.25    4.15    4.26    4.24    4.18    1.93    1.86    3.69    0.90
'''
