import os
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy

from surprise.model_selection import train_test_split

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

trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()
algo.fit(trainset, verbose = True, n_epochs = 5)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)