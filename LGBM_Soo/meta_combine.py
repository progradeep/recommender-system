import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import re
import requests, time
#
# tbcc_meta = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\Final_DATA\TBC_MOVIES_TITLE.xlsx")
#
# first_cr = pd.read_excel("C:\\Users\msi\Desktop\\naver_combined.xlsx")
#                          # dtype={'MOVIE_ID':np.uint32})
# second_cr = pd.read_excel("./cral_with_id.xlsx",dtype={'MOVIE_ID':np.uint32})
#
# tbcc_meta = tbcc_meta.merge(first_cr,how="left",on="MOVIE_ID")
# tbcc_meta['MOVIE_ID'] = tbcc_meta['MOVIE_ID'].astype(np.uint32)
#
#
#
# tbcc_meta = tbcc_meta.merge(second_cr,how="left",on="MOVIE_ID")
# print(tbcc_meta.columns)
# tbcc_meta['SUBTITLE_x'] = tbcc_meta['SUBTITLE_x'].fillna(tbcc_meta['SUBTITLE_y'])
# tbcc_meta['PUBYEAR_x'] = tbcc_meta['PUBYEAR_x'].fillna(tbcc_meta['PUBYEAR_y'])
# tbcc_meta['ACTOR_x'] = tbcc_meta['ACTOR_x'].fillna(tbcc_meta['ACTOR_y'])
# tbcc_meta['RATING_x'] = tbcc_meta['RATING_x'].fillna(tbcc_meta['RATING_y'])
# tbcc_meta['DIRECTOR_x'] = tbcc_meta['DIRECTOR_x'].fillna(tbcc_meta['DIRECTOR_y'])
# tbcc_meta['MOVIELENGTH_x'] = tbcc_meta['MOVIELENGTH_x'].fillna(tbcc_meta['MOVIELENGTH_y'])
# tbcc_meta['NUMRATING_x'] = tbcc_meta['NUMRATING_x'].fillna(tbcc_meta['NUMRATING_y'])
#
# tbcc_meta = tbcc_meta.drop(tbcc_meta.columns[tbcc_meta.columns.str.contains('_y')],axis=1)
# # train = train.drop(train.columns[train.columns.str.contains('unnamed',case=False)],axis=1)
# tbcc_meta.to_excel("FINAL.xlsx")
#
# print(tbcc_meta)

tbcc_meta = pd.read_excel('FINAL.xlsx')
tbcc_meta['SUBTITLE'] = tbcc_meta['SUBTITLE'].fillna('no subtitle')
tbcc_meta['SUBTITLE'] = tbcc_meta['SUBTITLE'].astype('category')

tbcc_meta['ACTOR'] = tbcc_meta['ACTOR'].fillna('no actor')
tbcc_meta['ACTOR'] = tbcc_meta['ACTOR'].astype('category')

tbcc_meta['DIRECTOR'] = tbcc_meta['DIRECTOR'].fillna('no director')
tbcc_meta['DIRECTOR'] = tbcc_meta['DIRECTOR'].astype('category')

tbcc_meta['PUBYEAR'] = tbcc_meta['PUBYEAR'].fillna(2000)
tbcc_meta['PUBYEAR'] = tbcc_meta['PUBYEAR'].astype(np.uint32)

tbcc_meta['RATING'] = tbcc_meta['RATING'].fillna(5.)
tbcc_meta['RATING'] = tbcc_meta['RATING'].astype(np.float32)

tbcc_meta['MOVIELENGTH'] = tbcc_meta['MOVIELENGTH'].fillna(60)
tbcc_meta['MOVIELENGTH'] = tbcc_meta['MOVIELENGTH'].astype(np.int32)

tbcc_meta['NUMRATING'] = tbcc_meta['NUMRATING'].fillna(0)
tbcc_meta['NUMRATING'] = tbcc_meta['NUMRATING'].astype(np.int32)


tbcc_meta.to_excel("FINAL.xlsx")