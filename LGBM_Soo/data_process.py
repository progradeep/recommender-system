import pandas as pd
import numpy as np
import re

data_path = "../../data/"
train = pd.read_csv(data_path + 'KISA_TBC_VIEWS_UNI.csv')
# user id, movie id, timestamp, duration, sequence
train = train[train.loc['USER_ID']>393186]


def sort(row):
    sorted_row = row.sort_values(by='duration', ascending=False)
    print(sorted_row)
    if sorted_row.shape[0] < 5:
        lack = 5 - sorted_row.shape[0]
        print(sorted_row['USER_ID'].values)
        out = [sorted_row['USER_ID'].values[0]] + list(sorted_row['movie id'].values) + [9000] * lack
        return out  # list(sorted_row['movie id']).extend([9000] * lack)

    out = [sorted_row['USER_ID'].values[0]] + list(sorted_row['movie id'].values)
    return out

grouped = train.groupby(by='USER_ID').apply(sort).tolist()
out = pd.DataFrame(grouped,columns=['USER_ID','MOVIE_1','MOVIE_2','MOVIE_3','MOVIE_4','MOVIE_5'])
print(out)
out.to_csv('top_5_duration_Q.csv')
