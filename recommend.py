import gzip
import json
from collections import namedtuple
import numpy as np
from lenskit import batch, topn, util,datasets
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, user_knn,item_knn
from lenskit import topn
import pandas as pd
import pickle

#Define the train-tesing pair
TTPair = namedtuple('TTPair', ['train', 'test'])
TTPair.__doc__ = 'Train-test pair (named tuple).'
TTPair.train.__doc__ = 'Train data for this pair.'
TTPair.test.__doc__ = 'Test data for this pair.'

#Split the dataset
def partition(data, partitions):
    rows = np.arange(len(data))
    test_sets = np.array_split(rows, partitions)

    for i, ts in enumerate(test_sets):
        print(i)
        print(ts)
        test = data.iloc[ts, :]
        trains = test_sets[:i] + test_sets[(i + 1):]
        train_idx = np.concatenate(trains)
        train = data.iloc[train_idx, :]
        yield TTPair(train, test)

#Make recommendations for top-10
def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # run the recommender
    recs = batch.recommend(fittable, users, 10)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

if __name__=="__main__":
    # with open('data/reviews5-2.pkl', 'rb') as f:
    #     train_set = pickle.load(f)
    #     test_set = pickle.load(f)
    # train_df = pd.DataFrame(train_set, columns=["user","item","rating"])
    # test_df = pd.DataFrame(test_set, columns=["user","item","rating"])
    # dataset = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    #
    #
    # train_test_pair=[]
    # for i, j in enumerate(partition(dataset, 5)):
    #     train_test_pair.append(j)
    #
    #
    # algo_ii = item_knn.ItemItem(20)
    # all_recs=eval('ItemItem', algo_ii, train_test_pair[4].train, train_test_pair[4].test)
    # all_recs.head()
    #
    # rla = topn.RecListAnalysis()
    # rla.add_metric(topn.ndcg)
    #
    # results = rla.compute(all_recs,train_test_pair[4].test)
    # results.head()
    with open('data/reviews5-1.pkl', 'rb') as f:
         dataset = pickle.load(f)

    algo_ii = item_knn.ItemItem(20)
    algo_als = als.BiasedMF(50)
    all_recs = []
    test_data = []
    for train, test in xf.partition_users(dataset[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        test_data.append(test)
        all_recs.append(eval('ItemItem', algo_ii, train, test))
        all_recs.append(eval('ALS', algo_als, train, test))

    all_recs = pd.concat(all_recs, ignore_index=True)
    print(all_recs.head())
    test_data = pd.concat(test_data, ignore_index=True)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    # print(results.head())