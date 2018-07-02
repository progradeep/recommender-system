import pandas as pd
import numpy as np
# meta = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\Movie_meta.xlsx")
# tbcc = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\TBCC_boxoffice.xlsx")
#
# titles = list(tbcc['TITLE'])
# out = {'GENRE':[],'DIRECTOR':[]}
#
# for m in titles:
#     if m in list(meta['TITLE']):
#         row = meta.loc[meta['TITLE'] == m]
#         out['GENRE'].append(row['GENRE'].values[0])
#         out['DIRECTOR'].append(row['DIRECTOR'].values[0])
#     else:
#
#         out['GENRE'].append("")
#         out['DIRECTOR'].append("")
#
# tbcc['GENRE'] = out['GENRE']
# tbcc['DIRECTOR'] = out['DIRECTOR']
#
# print(tbcc)
#
# tbcc.to_excel("tbcc_meta.xlsx")

tbcc = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\TBCC_meta.xlsx")
genres = list(tbcc['BOXOFFICE'])

print(len(genres))

genres = np.array(genres)
print(genres)

np.save("boxoffice.npy", genres)

# whole_dir = []
#
# for genre in genres:
#     genre = str(genre).split(',')
#     for g in genre:
#         if g in whole_dir or g == 'nan':
#             continue
#         else:
#             whole_dir.append(g)
#
#
# print(len(whole_dir))
#
# genre_onehot = np.zeros((8259, len(whole_dir)))
#
# for i, genre in enumerate(genres):
#     genre = str(genre).split(',')
#     for g in genre:
#         if g in whole_dir:
#             id = whole_dir.index(g)
#             genre_onehot[i,id] = 1.
#
# print(genre_onehot)
# np.save("director_onehot.npy", genre_onehot)