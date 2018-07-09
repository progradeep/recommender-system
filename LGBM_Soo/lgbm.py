import pandas as pd
import numpy as np

# data_path ="C:\\Users\msi\Downloads\\all\\"
#
# train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
#                                                 'source_system_tab' : 'category',
#                                                   'source_screen_name' : 'category',
#                                                   'source_type' : 'category',
#                                                   'target' : np.uint8,
#                                                   'song_id' : 'category'})
# test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
#                                                 'source_system_tab' : 'category',
#                                                 'source_screen_name' : 'category',
#                                                 'source_type' : 'category',
#                                                 'song_id' : 'category'})
# songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',
#                                                   'language' : 'category',
#                                                   'artist_name' : 'category',
#                                                   'composer' : 'category',
#                                                   'lyricist' : 'category',
#                                                   'song_id' : 'category'})
# members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category',
#                                                       'bd' : np.uint8,
#                                                       'gender' : 'category',
#                                                       'registered_via' : 'category'},
#                      parse_dates=['registration_init_time','expiration_date'])
# songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
#
# print("Done loading!")
#
# print("Data mergind start!")

# tbcc = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\TBC_MOVIES_TITLE.xlsx")
# db = pd.read_excel("C:\\Users\msi\Downloads\영화정보 리스트_2018-07-09.xlsx")
#
#
# tbcc_meta = tbcc.merge(db, on='TITLE', how='left')
#
# tbcc_meta.to_excel("./tbcc_meta.xlsx")
# tbcc = pd.read_excel("./tbcc_meta.xlsx")
# tbcc_new = tbcc.groupby('TITLE')
# # print(tbcc_new.first())
# def drop(row):
#     if len(row) > 1:
#         row = row.sort_values(['TYPE','MAKE_YEAR'], ascending=False)
#         row = row[:1]
#     return row
#
# tbcc_new = tbcc_new.apply(drop).sort_values(['MOVIE_ID'], ascending=True)
# print(tbcc_new)
# tbcc = pd.read_excel('./dropduplicates.xlsx')
#
# boxoffice = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\\boxoffice.xlsx")
# print(boxoffice)
# tbcc = tbcc.merge(boxoffice, on='TITLE', how='left')
# tbcc.to_excel("./tbcc_combined.xlsx")
p = pd.DataFrame({'a':[1,2,3]})
print(p['a'].shape)
a = np.ones(p['a'].shape)
print(a)