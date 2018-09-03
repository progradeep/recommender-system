#-*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# print("Loading train, test data")
data_path = "../../data/"

# train_pos = pd.read_csv(data_path+"KISA_TBC_VIEWS_UNIQ_TRAIN.csv",
#                         dtype={'USER_ID':'category', 'MOVIE_ID':'category'})
#
# train_neg = pd.read_csv(data_path+"KISA_TBC_NEG_TRAIN_SMALL.csv",
#                         dtype={'USER_ID':'category', 'MOVIE_ID':'category'})
#
# test = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',
#                    dtype={'USER_ID':'category', 'MOVIE_ID':'category'})
#
# train_pos['TARGET'] = 1.0
# train_neg['TARGET'] = 0.0
#
# train = pd.concat([train_pos,train_neg])
# train = train.sample(frac=1).reset_index(drop=True)
#
# del(train_pos)
# del(train_neg)
#
# train.to_csv(data_path+"train_tmp.csv")
# print("Train data:")
# print(train[:10])

print("Loading meta data")

user_pref = pd.read_csv(data_path+'user_pref.csv', dtype={'USER_ID':np.uint32, 'GENRE':'category', 'NATION':'category'})
mean_watch_count = pd.read_csv(data_path+"mean_watch_count.csv", dtype={'USER_ID':np.uint32,'MEAN_WATCH_COUNT':np.uint32})

"""
meta = pd.read_excel(data_path+"meta_combined.xlsx")

meta['TITLE'] = meta['TITLE'].astype('category')
meta['COUNTRY'] = meta['COUNTRY'].astype('category')
meta['TYPE'] = meta['TYPE'].astype('category')
meta['GENRE'] = meta['GENRE'].astype('category')
meta['DIRECTOR'] = meta['DIRECTOR'].astype('category')


# meta = meta.merge(watch_count,how='left',on='MOVIE_ID')

meta['MOVIE_ID'] = meta['MOVIE_ID'].astype(np.uint32)


meta['MAKE_YEAR'].fillna(2000, inplace=True)
meta['MAKE_YEAR'] = meta['MAKE_YEAR'].astype(np.uint16)

meta['COUNTRY'] = meta['COUNTRY'].cat.add_categories(['no_country'])
meta['COUNTRY'].fillna('no_country', inplace=True)

meta['TYPE'] = meta['TYPE'].cat.add_categories(['no_type'])
meta['TYPE'].fillna('no_type', inplace=True)

meta['GENRE'] = meta['GENRE'].cat.add_categories(['no_genre'])
meta['GENRE'].fillna('no_genre', inplace=True)

meta['DIRECTOR'] = meta['DIRECTOR'].cat.add_categories(['no_dir'])
meta['DIRECTOR'].fillna('no_dir', inplace=True)

meta['BOXOFFICE'].fillna(0, inplace=True)
meta['BOXOFFICE'] = meta['BOXOFFICE'].astype(np.uint32)

print(meta.dtypes)

print("Meta data:")
print(meta[:10])


# merge train and meta
tmp = pd.DataFrame(columns=['USER_ID','MOVIE_ID','TARGET'])
reader = pd.read_csv(data_path+'train_tmp.csv',dtype={'USER_ID':'category',
                                       'MOVIE_ID':np.uint32,
                                       'TARGET':np.float32},
                     chunksize=10000000)

train_key = tmp.columns
del(tmp)

#train = pd.DataFrame(columns=train_key.append(meta.columns).unique())

train['USER_ID'] = train['USER_ID'].astype(np.uint32)
train['MOVIE_ID'] = train['MOVIE_ID'].astype(np.uint32)


train['TITLE'] = train['TITLE'].astype('category')
train['MAKE_YEAR'] = train['MAKE_YEAR'].astype(np.uint16)
train['COUNTRY'] = train['COUNTRY'].astype('category')
train['TYPE'] = train['TYPE'].astype('category')
train['GENRE'] = train['GENRE'].astype('category')
train['DIRECTOR'] = train['DIRECTOR'].astype('category')
train['BOXOFFICE'] = train['BOXOFFICE'].astype(np.uint32)

train['WATCH_COUNT'] = train['WATCH_COUNT'].astype(np.uint32)


def preprocess_train(x):
    x['MOVIE_ID'] = x['MOVIE_ID'].astype(np.uint32)
    tmp_train = x.merge(meta, on='MOVIE_ID',how='left')
    return tmp_train


for r in reader:
    train = train.append(preprocess_train(r))
    print(train.shape)

print(train[:10])

"""

train = pd.read_csv(data_path+'train_tmp.csv',dtype={'USER_ID':np.uint32,
                                       'MOVIE_ID':np.uint32,
                                       'TARGET':np.uint32})


train = train.merge(user_pref, how='left',on='USER_ID')
train = train.merge(mean_watch_count,how='left',on='USER_ID')

train['MOVIE_ID'] = train['MOVIE_ID'].astype('category')
train['USER_ID'] = train['USER_ID'].astype('category')
train = train.drop(train.columns[train.columns.str.contains('unnamed',case=False)],axis=1)
print(train[:100])
train = train.drop(['USER_ID'],axis=1)

print("TRAIN")
print(train[:100])


# splitting test and train set
print("Splitting into train and val")

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')

X_train = train.drop(['TARGET'], axis=1)
y_train = train['TARGET'].values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val)
print('Processed data...')

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.1 ,
        'verbose': 0,
        'num_leaves': 200,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 300,
        'metric' : 'auc'
    }


lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)

del(train)
del(lgb_train)
del(lgb_val)

question_num = 810625237
batch_size = 10000000
total_step = question_num // batch_size + 1

# read test data
reader = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',
                   dtype={'USER_ID':np.uint32, 'MOVIE_ID':np.uint32},
                     chunksize=batch_size)

tmp = pd.DataFrame(columns=['USER_ID','MOVIE_ID'])

test_key = tmp.columns
del(tmp)


mean_watch_count = pd.read_csv(data_path+'mean_watch_count_Q.csv', dtype={'USER_ID':np.uint32, 'MEAN_WATCH_COUNT':np.uint32})

def preprocess_test(x):
    x['USER_ID'] = x['USER_ID'].astype(np.uint32)
    
    tmp_test = x.merge(user_pref,how='left',on='USER_ID')
    tmp_test = tmp_test.merge(mean_watch_count,how='left',on='USER_ID')
    
    tmp_test['MOVIE_ID'] = tmp_test['MOVIE_ID'].astype('category')
    tmp_test['USER_ID'] = tmp_test['USER_ID'].astype('category')
    tmp_test = tmp_test.drop(tmp_test.columns[tmp_test.columns.str.contains('unnamed', case=False)], axis=1)
    tmp_test = tmp_test.drop(['USER_ID'],axis=1)

    print(tmp_test.shape)
    print(tmp_test[:10])
    print(tmp_test[9000:9010])
    return tmp_test


subm = pd.DataFrame()
step = 0
for r in reader:
    #if step == 2: break
    predictions = lgbm_model.predict(preprocess_test(r))
    temp = pd.DataFrame()
    temp['target'] = predictions
    if step == 0:
        subm = temp
    else:
        subm = pd.concat([subm, temp])

    print('step: ' + str(step) + '/' + str(total_step))
    step += 1

print("SAVING..")

subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression='gzip', index=False, float_format='%.5f')

"""
# test = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',dtype={'USER_ID':'category',
#                                        'MOVIE_ID':'category'})
print("Load test data")

subm = pd.DataFrame()
reader = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',
                   dtype={'USER_ID':"category", 'MOVIE_ID':"category"},
                     chunksize=batch_size)

step = 0
for r in reader:
    #if step == 2: break
    predictions = lgbm_model.predict(r)
    temp = pd.DataFrame()
    temp['target'] = predictions
    if step == 0:
        subm = temp
    else:
        subm = pd.concat([subm, temp])
    step += 1
    print('step: ' + str(step) + '/' + str(total_step))

print("Saving...")
subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression='gzip', index=False, float_format='%.5f')
#
# for step in range(total_step):
#     if step == 2 : break
#     if step == total_step - 1:
#         test_batch = test[step * batch_size:]
#     else:
#         test_batch = test[step * batch_size:(step + 1) * batch_size]
#
#     predictions = lgbm_model.predict(test_batch)
#     temp = pd.DataFrame()
#     temp['target'] = predictions
#     if step == 0:
#         subm = temp
#     else:
#         subm = pd.concat([subm, temp])
#
#     print('step: ' + str(step) + '/' + str(total_step))
#
# subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression='gzip', index=False, float_format='%.5f')
"""
