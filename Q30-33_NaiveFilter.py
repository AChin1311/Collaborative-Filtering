
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import AlgoBase
from surprise.model_selection import cross_validate
from surprise.model_selection.split import train_test_split
import matplotlib.pyplot as plt

reader = Reader(sep=',')
data = Dataset.load_from_file('./ml-latest-small/ratings_new.csv', reader=reader)
data_pop = Dataset.load_from_file('./ml-latest-small/ratings_popular.csv', reader=reader)
data_unpop = Dataset.load_from_file('./ml-latest-small/ratings_unpopular.csv', reader=reader)
data_var = Dataset.load_from_file('./ml-latest-small/ratings_var.csv', reader=reader)


class NaiveFiltering(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        if self.trainset.knows_user(u):
            return np.mean([r for (_, r) in self.trainset.ur[u]])
        else:
            return self.trainset.global_mean


naive = NaiveFiltering()
print("Average RMSE for all data is {}".format(np.mean(cross_validate(naive, data, measures=['RMSE'], cv=10,  verbose=True)['test_rmse'])))
#0.9624544400980524

print("Average RMSE for popular movie trimmed data is {}".format(np.mean(cross_validate(naive, data_pop, measures=['RMSE'], cv=10,  verbose=True)['test_rmse'])))
#0.9585756614709473

print("Average RMSE for unpopular movie trimmed data is {}".format(np.mean(cross_validate(naive, data_unpop, measures=['RMSE'], cv=10,  verbose=True)['test_rmse'])))
#0.9789649538726051

print("Average RMSE for high variance movie trimmed data is {}".format(np.mean(cross_validate(naive, data_var, measures=['RMSE'], cv=10,  verbose=True)['test_rmse'])))
#0.9553821252342578
