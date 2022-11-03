from collections import namedtuple
import pandas as pd
import numpy as np
from surprise import Dataset, NormalPredictor, Reader,KNNBasic,accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from lenskit import batch, topn, util,datasets
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, user_knn,item_knn
from lenskit import topn

# Creation of the dataframe. Column names are irrelevant.

TTPair = namedtuple('TTPair', ['train', 'test'])
TTPair.__doc__ = 'Train-test pair (named tuple).'
TTPair.train.__doc__ = 'Train data for this pair.'
TTPair.test.__doc__ = 'Test data for this pair.'

def partition(data, partitions):
    rows = np.arange(len(data))
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        print(i)
        print(ts)
        test = data.iloc[ts, :]
        trains = test_sets[:i] + test_sets[(i + 1):]
        train_idx = np.concatenate(trains)
        train = data.iloc[train_idx, :]
        yield TTPair(train, test)




ratings_dict = {
    "user": [2, 2,3],
    "item": [ 45, 7,8],
    "rating": [ 3, 1,5],
    "timestamp":[2,6,8]
}

ratings_dict1 = {
    "user": [1, 2, 3,3,5,6,4,2, 2,3,3,5,6,4,3,5,3,3,3, 3],
    "item": [9, 7, 45,67,45,23,14,9, 32, 2,67,45,23,14,2,67,45,23,14,9],
    "rating": [3, 2, 4, 1,5,2,   3,1,2,3,   1,4,2,5,1,3,4,5,1,2],
        "timestamp":[7, 45,67,45,23,14,9, 2,6,8,3, 2, 4, 1,5,2,8,9,1,2]
}

train_test_pair=[]
df1 = pd.DataFrame(ratings_dict)
df2 = pd.DataFrame(ratings_dict1)
print(df1)
print(df2)
dataset=pd.concat([df1, df2], axis=0).reset_index(drop=True)
print(dataset)

for i,j in enumerate(partition(dataset,5)):
    print(i)
    print(j)
    train_test_pair.append(j)

print(train_test_pair[4])

# print(xf.partition_rows(dataset,1))
# for i, tp in enumerate(xf.partition_rows(dataset,1)):
#     print("The i is: ", i)
#     print("The tp is: ", tp)
    # print("lenthis:",len(tp))




# # A reader is still needed but only the rating_scale param is requiered.
# reader = Reader(rating_scale=(1, 5))
#
# # The columns must correspond to user id, item id and ratings (in that order).
# data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
# data1=data.build_full_trainset()
#
# # ghsyu, data2 = train_test_split(data, test_size=1)
# algo=KNNBasic()
# algo.fit(data1)
# # a=algo.test(data2)
# # accuracy.rmse(a)
