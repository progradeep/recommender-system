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
watch_count = pd.read_csv(data_path+"watch_count.csv")
watch_count.columns = ['MOVIE_ID',"WATCH_COUNT"]
watch_count = watch_count.astype(dtype={'MOVIE_ID':'category', 'WATCH_COUNT':np.uint32})

top5_duration = pd.read_csv(data_path+'top_5_duration.csv')
top5_duration = top5_duration.astype(dtype={'USER_ID':'category',
'1':'category','2':'category','3':'category','4':'category','5':'category'})

mean_watch_count = pd.read_csv(data_path+"mean_watch_count.csv")
mean_watch_count = mean_watch_count.astype(dtype={'USER_ID':'category','MEAN_WATCH_COUNT':np.uint32})


meta = pd.read_excel(data_path+"meta_combined.xlsx")

meta['MOVIE_ID'] = meta['MOVIE_ID'].astype('category')
meta['TITLE'] = meta['TITLE'].astype('category')
meta['COUNTRY'] = meta['COUNTRY'].astype('category')
meta['TYPE'] = meta['TYPE'].astype('category')
meta['GENRE'] = meta['GENRE'].astype('category')
meta['DIRECTOR'] = meta['DIRECTOR'].astype('category')


meta = meta.merge(watch_count,how='left',on='MOVIE_ID')

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

meta['WATCH_COUNT'].fillna(0, inplace=True)
meta['WATCH_COUNT'] = meta['WATCH_COUNT'].astype(np.uint32)

print(meta.dtypes)


print("Meta data:")
print(meta[:10])


# merge train and meta
tmp = pd.DataFrame(columns=['USER_ID','MOVIE_ID','TARGET'])
reader = pd.read_csv(data_path+'train_tmp.csv',dtype={'USER_ID':'category',
                                       'MOVIE_ID':np.uint32,
                                       'TARGET':np.float32},
                     chunksize=100000)

train_key = tmp.columns
del(tmp)

train = pd.DataFrame(columns=train_key.append(meta.columns).unique())
train.to_csv(data_path+"FINAL_train_merged.csv")

train['USER_ID'] = train['USER_ID'].astype('category')
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
    # tmp_train.to_csv(data_path+"FINAL_train_merged.csv",mode='a',index_label=False,header=None)


for r in reader:
    train = train.append(preprocess_train(r))
    print(train.shape)

print(train[:10])

# train = pd.read_csv(data_path+'FINAL_train_merged.csv')
# print(train.dtypes)

train = train.merge(top5_duration,how='left',on='USER_ID')
train = train.merge(mean_watch_count,how='left',on='USER_ID')

train['MOVIE_ID'] = train['MOVIE_ID'].astype('category')

print("Train data:")
print(train[:10])


# merge test and meta
reader = pd.read_csv(data_path+'KISA_TBC_NEG_QUESTION.csv',
                   dtype={'USER_ID':'category', 'MOVIE_ID':np.uint32},
                     chunksize=100000)

tmp = pd.DataFrame(columns=['USER_ID','MOVIE_ID'])

test_key = tmp.columns
del(tmp)

test = pd.DataFrame(columns=test_key.append(meta.columns).unique())

def preprocess_test(x,test):
    x['MOVIE_ID'] = x['MOVIE_ID'].astype(np.uint32)
    tmp_test = x.merge(meta, on='MOVIE_ID',how='left')
    train = test.append(tmp_test)

[preprocess_test(r,test) for r in reader]


test = test.merge(top5_duration,how='left',on='USER_ID')
test = test.merge(mean_watch_count,how='left',on='USER_ID')

test['MOVIE_ID'] = test['MOVIE_ID'].astype('category')

print("Test data:")
print(test[:10])

print("Merge finished")




# splitting test and train set
print("Splitting into train and val")

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

question_num = 810625237
batch_size = 1000000
total_step = question_num // batch_size + 1

subm = pd.DataFrame()
for step in range(total_step):
    if step == total_step - 1:
        test_batch = test[step * batch_size:]
    else:
        test_batch = test[step * batch_size:(step + 1) * batch_size]
    predictions = lgbm_model.predict(test_batch)

    temp = pd.DataFrame()
    temp['target'] = predictions
    if step == 0:
        subm = temp
    else:
        subm = pd.concat([subm, temp])

    print('step: ' + str(step) + '/' + str(total_step))

subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression='gzip', index=False, float_format='%.5f')
