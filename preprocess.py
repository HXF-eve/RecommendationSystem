import pickle
import pandas as pd
import gzip
import json
import random

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


#check for the missing value
def check_null_data(df):
    print(df.isnull().sum())
    # total = df.isnull().sum().sort_values(ascending=False)
    # percent =(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    # missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    # print(missing_data)

#dataset splitting: for each user,4:1 train_set:testing_set
def splitdataset_random(df):
    #sampling from the raw data
    # df = df.sample(frac=0.2, axis=0)
    train=[]
    test=[]
    for reviewerID, hist in df.groupby('user'):
        train_df=hist.sample(frac=0.8)
        test_df = hist.drop(train_df.index)
        #train.append(train_df.values.tolist())
        #test.append(test_df.values.tolist())
        for item in train_df.values.tolist():
            train.append(item)
        for item in test_df.values.tolist():
            test.append(item)
    random.shuffle(train)
    random.shuffle(test)
    print("The number of the training set is: ",len(train))
    print("The number of the testing set is: ", len(test))

    with open('data/reviews5-2.pkl', 'wb') as f:
        pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    reviews_df = getDF('data/reviews_Musical_Instruments_5.json.gz')
    reviews_df = reviews_df[['reviewerID', 'asin', 'overall']]
    reviews_df.columns=["user","item","rating"]
    for reviewerID, hist in reviews_df.groupby('user'):
        print(hist.values.tolist())
    #The number of the total users
    print("The number of the users is: ", len(reviews_df.groupby('user')))
    #The number of the total items
    print("The number of the items is: ", len(reviews_df.groupby('item')))
    #The number of the total ratings
    print("The number of the total ratings is: ", len(reviews_df))
    with open('data/reviews5-1.pkl', 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    splitdataset_random(reviews_df)
