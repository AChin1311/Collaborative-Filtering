import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import AlgoBase
from surprise.model_selection import cross_validate
from surprise.model_selection.split import train_test_split
import matplotlib.pyplot as plt
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.matrix_factorization import SVD
plt.close('all')


reader = Reader(sep=',')
data = Dataset.load_from_file('./ml-latest-small/ratings_new.csv', reader=reader)

data.split(n_folds=10)
sim_options = {'name': 'pearson',              'user_based': True              }
algo1 = KNNWithMeans(k=48, sim_options=sim_options)
algo2 = NMF(n_factors=16)
algo3 = SVD(n_factors=14)


def RankSweep(algo, tit, num):
    t_all = range(1, 26)
    pre_all = np.zeros(25)
    rec_all = np.zeros(25)
    for trainset, testset in data.folds():
        algo.fit(trainset)
        pred = algo.test(testset)
        G_all = dict()
        S_all = dict()
        for elem in pred:
            if elem.r_ui >= 3:
                G_all.setdefault(elem.uid, set()).add(elem.iid)
            S_all.setdefault(elem.uid, list()).append((elem.est, elem.iid))
        pre = [[] for _ in range(25)]
        rec = [[] for _ in range(25)]

        for uid, G in G_all.items():
            S = sorted(S_all[uid], key = lambda x: x[0], reverse = True)
            inter = 0
            for t in t_all:
                if len(S) < t:
                    break
                if S[t - 1][1] in G:
                    inter += 1
                pre[t - 1].append(inter / t)
                rec[t - 1].append(inter / len(G))

        for i in range(25):
            pre_all[i] += np.mean(pre[i])
            rec_all[i] += np.mean(rec[i])

    pre_all /= 10
    rec_all /= 10
    plt.figure()
    plt.plot(t_all, pre_all)
    plt.xlabel('t')
    plt.ylabel('precision')
    plt.title('precision vs t for {}'.format(tit))
    plt.savefig('plot/Q{}_precision_t.png'.format(num))
    plt.show()
    
    plt.figure()
    plt.plot(t_all, rec_all)
    plt.xlabel('t')
    plt.ylabel('recall')
    plt.title('recall vs t for {}'.format(tit))
    plt.savefig('plot/Q{}_recall_t.png'.format(num))
    plt.show()
    
    plt.figure()    
    plt.plot(rec_all, pre_all)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision vs recall for {}'.format(tit))
    plt.savefig('plot/Q{}_precision_recall.png'.format(num))
    plt.show()
    return (rec_all, pre_all)

r_p1 = RankSweep(algo1, 'KNN', 36)
r_p2 = RankSweep(algo2, 'NNMF', 37)
r_p3 = RankSweep(algo3, 'MF with bias', 38)

plt.figure()
plt.plot(*r_p1)
plt.plot(*r_p2)
plt.plot(*r_p3)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['KNN', 'NNMF', 'MF with bias'], loc='upper right')
plt.title('precision vs recall for 3 methods')
plt.savefig('plot/Q39_precision_recall.png')
plt.show()



