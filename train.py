import pickle
import pandas as pd
from surprise import Reader,Dataset,KNNBasic,accuracy,SVD,SVDpp,NMF,SlopeOne,CoClustering
from surprise.model_selection import train_test_split

#pre-processing for the prediction
def predict():
    with open('data/reviews5-2.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    train_df = pd.DataFrame(train_set, columns=["userID", "itemID", "rating"])
    test_df = pd.DataFrame(test_set, columns=["userID", "itemID", "rating"])
    reader = Reader(rating_scale=(1, 5))
    dataset = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    # dataset['rating']= dataset['rating'].astype('float32')
    data_set = Dataset.load_from_df(dataset, reader)
    trainset, testset = train_test_split(data_set, test_size=0.2, shuffle=False)
    return trainset,testset

#Different methods for predicting
def collaborativeFiltering(tr,te):
    #user_based,different similarity methods
    sim_options1={
        "name":"cosine",
        "user_based": True,
    }

    sim_options2 = {
        "name": "msd",
        "user_based": True,
    }

    sim_options3 = {
        "name": "pearson",
        "user_based": True,
    }
    #item-based
    sim_options4={
        "name": "cosine",
        "user_based": False,
    }

    sim_options5 = {
        "name": "msd",
        "user_based": False,
    }

    sim_options6 = {
        "name": "pearson",
        "user_based": False,
    }

    algo=KNNBasic(k=40, min_k=1,sim_options=sim_options1)
    algo.fit(tr)

    #uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
    #iid = str(302)  # raw item id (as in the ratings file). They are **strings**!
    # get a prediction for specific users and items.
    #pred = algo.predict(uid, iid, r_ui=4, verbose=True)

    preditions=algo.test(te)
    mae=accuracy.mae(preditions)
    rmse=accuracy.rmse(preditions)


def matrixFactorization(tr,te):
    algo = SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02, verbose=True)
    #A collaborative filtering algorithm based on Non-negative Matrix Factorization.
    # algo=NMF(n_factors=15, n_epochs=50, biased=False, reg_pu=0.06, reg_qi=0.06, reg_bu=0.02, reg_bi=0.02, lr_bu=0.005,
    #          lr_bi=0.005, init_low=0, init_high=1, random_state=None, verbose=True)
    algo.fit(tr)
    preditions = algo.test(te)
    mae = accuracy.mae(preditions)
    rmse = accuracy.rmse(preditions)

def slopeOne(tr,te):
    algo=SlopeOne()
    algo.fit(tr)
    preditions = algo.test(te)
    mae = accuracy.mae(preditions)
    rmse = accuracy.rmse(preditions)

def coClustering(tr,te):
    algo = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=20,verbose=True)
    algo.fit(tr)
    preditions = algo.test(te)
    mae = accuracy.mae(preditions)
    rmse = accuracy.rmse(preditions)

if __name__ == '__main__':
    tr,te=predict()
    collaborativeFiltering(tr,te)








