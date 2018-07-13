import pandas as pd
import numpy as np
import re


data_path = "../../data/"
with open(data_path+"ae_total.csv",'r') as f:
    watched_movie = f.readlines()
print(watched_movie)
watched_movie = watched_movie[393186:]

# user 1: movie 1, movie 2, movie 3, ... ,movie N

print(watched_movie)

watch_count = pd.read_csv(data_path+"watch_count.csv", header=None)
watch_count.columns = ['MOIVE_ID',"WATCH_COUNT"]
watch_count = watch_count.astype(dtype={'MOVIE_ID':'category', 'WATCH_COUNT':np.uint32})
# movie 1, count
# movie 2, count

watch_count = watch_count.sort_values(by='MOVIE_ID',ascending=True)
print(watch_count)

output = {}
for userId in range(watched_movie.shape[0]):
    # iterate over user ids
    w_movies = np.array(watched_movie[userId:userId+1].dropna(axis='columns'),dtype=int).squeeze()
    print(w_movies)
    w_count = watch_count.iloc[w_movies]
    print(w_count)
    output[userId] = w_count['WATCH_COUNT'].mean()
    print(output)

out = pd.DataFrame(output,columns=["USER_ID",'MEAN_COUNT'])
out.to_csv(data_path+'mean_watch_count_Q.csv')



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
out.to_csv(data_path+'top_5_duration_Q.csv')
