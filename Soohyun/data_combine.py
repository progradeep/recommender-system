import pandas as pd

meta = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\Movie_meta.xlsx")
tbcc = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\TBCC_boxoffice.xlsx")

titles = list(tbcc['TITLE'])
out = {'GENRE':[],'DIRECTOR':[]}

for m in titles:
    if m in list(meta['TITLE']):
        row = meta.loc[meta['TITLE'] == m]
        out['GENRE'].append(row['GENRE'].values[0])
        out['DIRECTOR'].append(row['DIRECTOR'].values[0])
    else:

        out['GENRE'].append("")
        out['DIRECTOR'].append("")

tbcc['GENRE'] = out['GENRE']
tbcc['DIRECTOR'] = out['DIRECTOR']

print(tbcc)

tbcc.to_excel("tbcc_meta.xlsx")