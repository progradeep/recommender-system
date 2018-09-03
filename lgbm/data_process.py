import pandas as pd
import numpy as np
import re
#
#
# data_path = "../../data/"
# with open(data_path+"ae_total.csv",'r') as f:
#     watched_movie = f.readlines()
# watched_movie = watched_movie[393187:]
#
# # user 1: movie 1, movie 2, movie 3, ... ,movie N
# watched_movie = [l.strip().split(',') for l in watched_movie]
#
# watched_movie = pd.DataFrame(watched_movie)
#
# print(watched_movie[:10])
#
# watch_count = pd.read_csv(data_path+"watch_count.csv")
# print(watch_count)
#
# watch_count = watch_count.astype(dtype={'MOVIE_ID':'category', 'WATCH_COUNT':np.uint32})
# # movie 1, count
# # movie 2, count
#
# print(watch_count)
#
# total_count = np.arange(8259)
# total = pd.DataFrame(total_count)
# total.columns = ['MOVIE_ID']
# total['WATCH_COUNT'] = 0.0
# # df.loc[df['A'] == 'foo']
#
# for w in range(watch_count.shape[0]):
#     w = watch_count[w:w+1]
#     try:
#         mid = int(w['MOVIE_ID'])
#     except: continue
#     wc = int(w['WATCH_COUNT'])
#
#     total.loc[mid,'WATCH_COUNT'] = wc
# print(total)
#
# output = []
# for userId in range(len(watched_movie)):
#     # iterate over user ids
#     w_movies = np.array(watched_movie[userId:userId+1].dropna(axis='columns'),dtype=int)[0] # user i's watched movie
#     print(userId,w_movies,type(w_movies))
#     w_count = total.loc[w_movies]
#     output.append([userId+393187, int(w_count['WATCH_COUNT'].mean())])
# out = pd.DataFrame(output)
# print(out)
# out.to_csv(data_path+'mean_watch_count_Q.csv')
#
#
#
# train = pd.read_csv(data_path + 'KISA_TBC_VIEWS_UNI.csv')
# # user id, movie id, timestamp, duration, sequence
# train = train[train.loc['USER_ID']>393186]
#
#
# def sort(row):
#     sorted_row = row.sort_values(by='duration', ascending=False)
#     print(sorted_row)
#     if sorted_row.shape[0] < 5:
#         lack = 5 - sorted_row.shape[0]
#         print(sorted_row['USER_ID'].values)
#         out = [sorted_row['USER_ID'].values[0]] + list(sorted_row['movie id'].values) + [9000] * lack
#         return out  # list(sorted_row['movie id']).extend([9000] * lack)
#
#     out = [sorted_row['USER_ID'].values[0]] + list(sorted_row['movie id'].values)
#     return out
#
# grouped = train.groupby(by='USER_ID').apply(sort).tolist()
# out = pd.DataFrame(grouped,columns=['USER_ID','MOVIE_1','MOVIE_2','MOVIE_3','MOVIE_4','MOVIE_5'])
# print(out)
# out.to_csv(data_path+'top_5_duration_Q.csv')

org = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\Final_DATA\\TBC_MOVIES_TITLE.xlsx")
naver = pd.read_excel("C:\\Users\msi\Desktop\\naver_crawled.xlsx")

print(org.dtypes)
print(naver.dtypes)

combine = org.merge(naver, on='MOVIE_ID',how = 'left')
print(combine)

combine.to_excel('naver_combined.xlsx')