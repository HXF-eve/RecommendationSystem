# RecommendationSystem
This is for Fall2022:Frontier of Cross-media Recommendation System(3131101531)–Project 

It contains three parts:

1.Raw data pre-processing

2.Rating prediction and error calculation

3.Recommendation and evaluation


Using the Amazon dataset(Digital_Instrument and Software), LenKit(http://lenskit.org/index.html)  and Surprise(https://github.com/NicolasHug/Surprise) toolkit.

preprocess.py:
It first reads the dataset from the raw file and uses a dataframe with columns["user","item","rating"] to represent the ratings. Then for each user, the raings is splitted into two parts including training dataset and testing dataset(4:1). We then visualize the training dataset and the testing dataset to show the distribution of the whole dataset.


train.py:
We use the training dataset to fit the model and use the testing dataset for evaluating the models/algorithms.
We provide several methods for rating prediction:
(1).Collaborative Filtering, including user-based and item-based CF approaches. Different  similarity calculation methods are applied include MSD(Mean Squared Difference), cosine similarity and the pearson correlation coefficient.
(2).Matrix Factorization, which used SVD algorithms.
(3).SlopOne, a simple but with high-accruracy method
(4).Coclustering.
We evaluate these algorithms by calculating the MAE and RMSE.
