import pandas as pd
import numpy as np

data_path = "../../data/"

""" # 4th todo
watched_movie = pd.read_csv(data_path+"ae_total.csv", dtype='category')
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
out.to_csv(data_path+'mean_watch_count.csv')
"""


"""
# 5. 전체 user 에 대해서 duration 순으로 상위 5개의 movie_id 기록하는 csv 만들기.
# 5개의 기록이 없으면 뒤에는 일단 9000으로 채워넣기

# train = pd.read_csv(data_path + 'KISA_VIEWS_TBC.csv')
# user id, movie id, timestamp, duration, sequence

train = pd.DataFrame([[0, 2, 2018, 60, 1],[0, 54, 2017, 47, 2], [1, 2, 2018, 2, 1], [1, 4, 2018, 21, 1],[1, 14, 2018, 32, 1]],
                     columns=["USER_ID", "movie id", "timestamp", "duration", "sequence"],dtype=int)

def sort(row):
    sorted_row = row.sort_values(by='duration',ascending=False)
    print(sorted_row)
    if sorted_row.shape[0] < 5:
        lack = 5 - sorted_row.shape[0]
        print(sorted_row['USER_ID'].values)
        out = [sorted_row['USER_ID'].values[0]]+list(sorted_row['movie id'].values) + [9000] * lack
        return out #list(sorted_row['movie id']).extend([9000] * lack)

    out = [sorted_row['USER_ID'].values[0]] + list(sorted_row['movie id'].values)
    return out

grouped = train.groupby(by='USER_ID').apply(sort).tolist()


out = pd.DataFrame(grouped,columns=['USER_ID','MOVIE_1','MOVIE_2','MOVIE_3','MOVIE_4','MOVIE_5'])
print(out)
out.to_csv('top_5_duration.csv')
"""

