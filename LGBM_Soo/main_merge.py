#-*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print("Loading train, test data")
data_path = "../../data/"

train_pos = pd.read_csv(data_path+"KISA_TBC_VIEWS_UNIQ_TRAIN.csv",
                        dtype={'USER_ID':'category', 'MOVIE_ID':'category'})

train_neg = pd.read_csv(data_path+"KISA_TBC_NEG_TRAIN_SMALL.csv",
                        dtype={'USER_ID':'category', 'MOVIE_ID':'category'})

test = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',
                   dtype={'USER_ID':'category', 'MOVIE_ID':'category'})

train_pos['TARGET'] = 1.0
train_neg['TARGET'] = 0.0

train = pd.concat([train_pos,train_neg])
train = train.sample(frac=1).reset_index(drop=True)

del(train_pos)
del(train_neg)

train.to_csv("../../train_tmp.csv")
print("Train data:")
print(train[:10])

print("Loading meta data")
watch_count = pd.read_csv(data_path+"watch_count.csv", header=None)
watch_count.columns = ['MOVIE_ID',"WATCH_COUNT"]
watch_count = watch_count.astype(dtype={'MOVIE_ID':'category', 'WATCH_COUNT':np.uint32})

top5_duration = pd.read_csv(data_path+'top_5_duration.csv', header=None)
top5_duration.columns = ['MOVIE_ID','1','2','3','4','5']
top5_duration = top5_duration.astype(dtype={'MOVIE_ID':'category',
'1':'category','2':'category','3':'category','4':'category','5':'category'})

mean_watch_count = pd.read_csv(data_path+"mean_watch_count.csv",dtype={'USER_ID':'category',
                                                                       'MEAN_WATCH_COUNT':np.uint32})



meta = pd.read_excel(data_path+"meta_combined.xlsx")


meta = meta.merge(watch_count,how='left',on='MOVIE_ID')
meta = meta.merge(top5_duration,how='left',on='MOVIE_ID')
meta = meta.merge(mean_watch_count,how='left',on='MOVIE_ID')

print("Meta data:")
print(meta[:10])

train_key = train.columns

del(train)

train = pd.DataFrame(columns=train_key.append(meta.columns).unique())
test = pd.DataFrame(columns=train_key.append(meta.columns).unique())


def preprocess_train(x):
    tmp_train = x.merge(meta, on='m',how='left')
    print(tmp_train)
    train = pd.concat([train,tmp_train])

reader = pd.read_csv(data_path+'train_tmp.csv',dtype={'u':'category',
                                       'm':'category',
                                       't':np.float32},
                     chunksize=100000)

[preprocess_train(r) for r in reader]
print(train)

def preprocess_test(x):
    tmp_test = x.merge(meta, on='m',how='left')
    print(tmp_test)
    test = pd.concat([test,tmp_test])

reader = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',
                   dtype={'USER_ID':'category', 'MOVIE_ID':'category'},
                     chunksize=100000)

[preprocess_test(r) for r in reader]
print(test)


print("merge finished")



# on train data
train['MAKE_YEAR'].fillna(2000, inplace=True)
train['MAKE_YEAR'] = train['MAKE_YEAR'].astype(np.uint16)

train['COUNTRY'] = train['COUNTRY'].cat.add_categories['no_country']
train['COUNTRY'].fillna('no_country', inplace=True)

train['TYPE'] = train['TYPE'].cat.add_categories['no_type']
train['TYPE'].fillna('no_type', inplace=True)

train['GENRE'] = train['GENRE'].cat.add_categories['no_genre']
train['GENRE'].fillna('no_genre', inplace=True)

train['DIRECTOR'] = train['DIRECTOR'].cat.add_categories['no_type']
train['DIRECTOR'].fillna('no_type', inplace=True)

train['BOXOFFICE'].fillna(0, inplace=True)
train['BOXOFFICE'] = train['BOXOFFICE'].astype(np.uint32)

train['WATCH_COUNT'].fillna(0, inplace=True)
train['WATCH_COUNT'] = train['WATCH_COUNT'].astype(np.uint32)

# on test data
test['MAKE_YEAR'].fillna(2000, inplace=True)
test['MAKE_YEAR'] = test['MAKE_YEAR'].astype(np.uint16)

test['COUNTRY'] = test['COUNTRY'].cat.add_categories['no_country']
test['COUNTRY'].fillna('no_country', inplace=True)

test['TYPE'] = test['TYPE'].cat.add_categories['no_type']
test['TYPE'].fillna('no_type', inplace=True)

test['GENRE'] = test['GENRE'].cat.add_categories['no_genre']
test['GENRE'].fillna('no_genre', inplace=True)

test['DIRECTOR'] = test['DIRECTOR'].cat.add_categories['no_type']
test['DIRECTOR'].fillna('no_type', inplace=True)

test['BOXOFFICE'].fillna(0, inplace=True)
test['BOXOFFICE'] = test['BOXOFFICE'].astype(np.uint32)

test['WATCH_COUNT'].fillna(0, inplace=True)
test['WATCH_COUNT'] = test['WATCH_COUNT'].astype(np.uint32)



print(train[:10])

# splitting test and train set
print("Splitting into train and test")

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

X_train = train.drop(['TARGET'], axis=1)
y_train = train['TARGET'].values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val)
print('Processed data...')

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.2 ,
        'verbose': 0,
        'num_leaves': 100,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 40,
        'metric' : 'auc'
    }


lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)


predictions = lgbm_model.predict(test)

# Writing output to file
subm = pd.DataFrame()
subm['target'] = predictions
subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
