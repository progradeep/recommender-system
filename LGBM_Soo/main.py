import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print("Loading data")
data_path = "../../data/"

train = pd.read_csv(data_path+"KISA_TBC_VIEWS_UNIQ.csv", dtype={'USER_ID':'category',
                                                                'MOVIE_ID':'category',
                                                                'DURATION':np.uint8,
                                                                'WATCH_DAY':np.uint16,
                                                                'WATCH_SEQ':np.uint8})

test = pd.read_csv(data_path+'question.csv',dtype={'USER_ID':'category',
                                                   'MOVIE_ID':'category',
                                                   'DURATION':np.uint8,
                                                   'WATCH_DAY':np.uint16,
                                                   'WATCH_SEQ':np.uint8
                                                   })

watch_count = pd.read_csv(data_path+"watch_count", header=None)
watch_count.columns = ['MOIVE_ID',"WATCH_COUNT"]
watch_count = watch_count.astype(dtype={'MOVIE_ID':'category', 'WATCH_COUNT':np.uint32})



meta = pd.read_csv(data_path+"meta_combined.csv",dtype={'MOVIE_ID':'category',
                                                        'TITLE':'category',
                                                        'MAKE_YEAR':np.uint16,
                                                        'COUNTRY':'category',
                                                        'TYPE':'category',
                                                        'GENRE':'category',
                                                        'DIRECTOR':'category',
                                                        'BOXOFFICE':np.uint32})


train = train.merge(meta, on='MOVIE_ID', how='left')
train = train.merge(watch_count, on='MOVIE_ID', how='left')

test = test.merge(meta, on='MOVIE_ID', how='left')
test = test.merge(watch_count, on='MOVIE_ID', how='left')

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


# adding new features

def country_bool(c):
    if u'한국' in c or u'미국' in c:
        return 1
    else: return 0

train['COUNTRY_BOOL'] = train['COUNTRY'].apply(country_bool).astype(np.int8)


# splitting test and train set
print("Splitting into train and test")

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

y_train = np.ones(train['USER_ID'].shape)

X_tr, X_val, y_tr, y_val = train_test_split(train, y_train)

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
        'num_rounds': 500,
        'metric' : 'auc'
    }

lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)

predictions = lgbm_model.predict(test)

# Writing output to file
subm = pd.DataFrame()
subm['target'] = predictions
subm.to_csv(data_path + 'lgbm_submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')