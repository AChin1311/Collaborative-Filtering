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

def plot_roc(y_true, y_estimate, threshold):
	fpr, tpr, _ = metrics.roc_curve(y_true, y_estimate)
	roc_auc = auc(fpr,tpr)
	plt.figure()
	plt.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
	plt.grid(color='0.7', linestyle='--', linewidth=1)
	plt.xlim([-0.1, 1.1])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate',fontsize=15)
	plt.ylabel('True Positive Rate',fontsize=15)

	plt.legend(loc = "lower right")
	plt.savefig('plot/q29_mf_roc_' + str(threshold) + '.png')
	plt.clf()

if __name__ == "__main__":
	threshold = [2.5, 3, 3.5, 4]
	file_path = os.path.expanduser("ml-latest-small/ratings_new.csv")
	reader = Reader(sep=',')
	data = Dataset.load_from_file(file_path, reader=reader)

	sim_options = {'name': 'pearson',
	              'user_based': True
	              }

	trainset, testset = train_test_split(data, test_size=0.1)

	for th in threshold:
		algo = SVD(n_factors=14)
		algo.fit(trainset)
		predictions = algo.test(testset)

		y_true = []
		y_estimate = []

		for row in predictions:
			if row[2] >= th:
				y_true.append(1)
			else:
				y_true.append(0)
			y_estimate.append(row[3])

		plot_roc(y_true,y_estimate, th)



