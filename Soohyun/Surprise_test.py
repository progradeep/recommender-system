import os
import pandas as pd
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy

from surprise.model_selection import train_test_split




def get_top_n(predictions, testdf, n=50):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if int(uid) < 393186: print(uid)
        if iid not in testdf.loc[testdf['USER_ID']==uid]['MOVIE_ID'] and int(uid) > 393186:
            top_n[uid].append((iid, est))
            print(uid)

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# path to dataset file
file_path = '../../data/KISA_TBC_VIEWS_UNIQ.csv'

df = pd.read_csv(file_path)

df = df.drop(columns = ['DURATION','WATCH_DAY','WATCH_SEQ'])
df['RATING'] = [1] * len(df['USER_ID'])

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_df(df, reader=reader)
trainset, testset = train_test_split(data, test_size=0.25, shuffle=False)

test_df = df[df['USER_ID'].gt(393186)]
# data = Dataset.load_from_df(test_df, reader=reader)
# testset = data.build_full_trainset()

# We'll use the famous SVD algorithm.
algo = SVD(verbose = True, n_epochs = 5)
algo.fit(trainset)
predictions = algo.test(testset, verbose=True)

# Then compute RMSE
# accuracy.rmse(predictions)
top_n = get_top_n(predictions, test_df, n=50)

output = pd.DataFrame(top_n)
output.to_csv("./output.csv")