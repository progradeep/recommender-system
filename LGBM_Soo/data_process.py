import pandas as pd
import numpy as np
import csv

data_path = "../../data/"
#
#  # 4th todo
"""
w = []
with open("../../data/ae_total.csv","rb") as f:
    watched_movie = f.readlines()
    #watched = csv.reader(f,delimiter=',')
    #w.append(watched)
    watched_movie = [l.strip().split(',') for l in watched_movie]
watched_movie = pd.DataFrame(watched_movie)[:393187]
print(watched_movie)
# # user 1: movie 1, movie 2, movie 3, ... ,movie N
#
# print(watched_movie)
#
watch_count = pd.read_csv(data_path+"watch_count.csv", header=None)
watch_count.columns = ['MOVIE_ID',"WATCH_COUNT"]
# # movie 1, count
# # movie 2, count
#
total_watch = pd.DataFrame(np.arange(8259), columns=['MOVIE_ID'])

total_watch['WATCH_COUNT'] = np.zeros(8259)

for i in range(watch_count.shape[0]):
    row = watch_count[i:i+1]
    movie_id = row['MOVIE_ID'].values[0]
    try: total_watch.loc[int(movie_id),'WATCH_COUNT'] = row['WATCH_COUNT'].values[0]
    except: 
        print(movie_id)
        continue
print(total_watch)
total_watch.to_csv('./test.csv')
output = {}
for userId in range(watched_movie.shape[0]):
    # iterate over user ids
    w_movies = np.array(watched_movie[userId:userId+1].dropna(axis='columns'),dtype=int).squeeze()
    print(userId)
    w_count = total_watch.iloc[w_movies]
    output[userId] = w_count['WATCH_COUNT'].mean().astype(int)

out = pd.DataFrame(output.items(),columns=['USER_ID','MEAN_WATCH_COUNT'])
out.to_csv(data_path+'mean_watch_count.csv')

"""



train = pd.read_csv(data_path + 'KISA_TBC_VIEWS_UNIQ.csv')
# user id, movie id, timestamp, duration, sequence

def sort(row):
    sorted_row = row.sort_values(by='DURATION',ascending=False)
    print(sorted_row.shape)
    if sorted_row.shape[0] < 5:
        lack = 5 - sorted_row.shape[0]
        print(sorted_row['USER_ID'].values[0], lack)
        out = list(sorted_row['MOVIE_ID'].values) + [8259] * lack
        return out #list(sorted_row['movie id']).extend([9000] * lack)

    out =  list(sorted_row['MOVIE_ID'].values[:5])
    return out

grouped = train.groupby(by='USER_ID').apply(sort).tolist()


#out = pd.DataFrame(grouped,columns=['USER_ID','MOVIE_1','MOVIE_2','MOVIE_3','MOVIE_4','MOVIE_5'], index=False)
out = pd.DataFrame(grouped)
print(out)
out.to_csv(data_path+'top_5_duration.csv')

