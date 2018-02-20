import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection.split import train_test_split
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.similarities import pearson
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc

if __name__ == "__main__":
	threshold = 3
	file_path = os.path.expanduser("ml-latest-small/ratings_new.csv")
	reader = Reader(sep=',')
	data = Dataset.load_from_file(file_path, reader=reader)

	sim_options = {'name': 'pearson',
	              'user_based': True
	              }

	trainset, testset = train_test_split(data, test_size=0.1)

	algo = KNNWithMeans(k=34, sim_options=sim_options)
	algo.fit(trainset)
	predictions1 = algo.test(testset)

	algo = NMF(n_factors=16)
	algo.fit(trainset)
	predictions2 = algo.test(testset)

	algo = SVD(n_factors=14)
	algo.fit(trainset)
	predictions3 = algo.test(testset)

	y_true = []
	y_estimate1 = []
	y_estimate2 = []
	y_estimate3 = []

	for row in predictions1:
		if row[2] >= threshold:
			y_true.append(1)
		else:
			y_true.append(0)
	
	for row in predictions1:
		y_estimate1.append(row[3])
	for row in predictions2:
		y_estimate2.append(row[3])
	for row in predictions3:
		y_estimate3.append(row[3])

	fpr1, tpr1, _ = metrics.roc_curve(y_true, y_estimate1)
	fpr2, tpr2, _ = metrics.roc_curve(y_true, y_estimate2)
	fpr3, tpr3, _ = metrics.roc_curve(y_true, y_estimate3)
	roc_auc1 = auc(fpr1,tpr1)
	roc_auc2 = auc(fpr2,tpr2)
	roc_auc3 = auc(fpr3,tpr3)
	plt.figure()
	plt.plot(fpr1, tpr1, lw=2, label= 'area under curve = %0.4f' % roc_auc1)
	plt.grid(color='0.2', linestyle='--', linewidth=1)
	plt.plot(fpr2, tpr2, lw=2, label= 'area under curve = %0.4f' % roc_auc2)
	plt.grid(color='0.5', linestyle='--', linewidth=1)
	plt.plot(fpr3, tpr3, lw=2, label= 'area under curve = %0.4f' % roc_auc3)
	plt.grid(color='0.7', linestyle='--', linewidth=1)
	plt.xlim([-0.1, 1.1])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate',fontsize=15)
	plt.ylabel('True Positive Rate',fontsize=15)
	plt.legend(loc = "lower right")


	plt.savefig('plot/all_roc_' + str(threshold) + '.png')
	plt.clf()



