#Importing Libraries

import gc
import os
from pathlib import Path
import time

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import f1_score
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
import warnings
warnings.simplefilter('ignore')

def reduce_mem_usage(df, use_float16=False):
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    from pandas.api.types import is_categorical_dtype
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def apply_label_encode(df, subject_cols):
    for str_col in subject_cols:
        # ===== assumes Series of string =====
        temp_dict = {value: i for i, value in enumerate(df[str_col].unique())}
        df[str_col] = df[str_col].map(temp_dict)
    del temp_dict, str_col; gc.collect()

def apply_freq_encode(df, str_col):
    temp_dict = {sample: df.loc[df[str_col]==sample].shape[0] for sample in df[str_col].unique()}
    #for sample in df[str_col].unique():
    #    temp_dict[sample] = df.loc[df[str_col]==sample].shape[0]
    df[f'{str_col}_ratio'] = df[str_col].map(temp_dict).astype(np.int32) / df[str_col].shape[0]
    del temp_dict; gc.collect()

def apply_cyclical(df, str_col):
    # df["hour"] = df["timestamp"].dt.hour
    # ===== assumes integer array =====
    df[str_col] -= df[str_col].min()
    int_max = df[str_col].max()
    df[f'{str_col}_sin'] = np.sin(2 * np.pi * df[str_col] / int_max)
    df[f'{str_col}_cos'] = np.cos(2 * np.pi * df[str_col] / int_max)
    del int_max; gc.collect()

def apply_target_encode(df, target_col, cat_col, smooth=False, m=0, statistic=False, print_option=True):
    # df[target_col] = np.log1p(df[target_col])
    df_group = df.groupby(cat_col)[target_col]
    group_mean = df_group.mean().astype(np.float16)
    
    # ===== smoothing =====
    if smooth:
        global_mean = df[target_col].mean()
        group_count = df_group.count().astype(np.float16)
        smoother = ((group_count * group_mean) + (m * global_mean)) / (group_count + m)
        df[f'{cat_col}_smoothed_mean'] = df[f'{cat_col}'].map(smoother)
        del global_mean, group_count, smoother; gc.collect()

    # ===== no smoothing =====
    elif smooth == False:
        df[f'{cat_col}_mean'] = df[f'{cat_col}'].map(group_mean)
        
    # ===== more target statistic =====
    if statistic:
        group_min = df_group.min().astype(np.float16)
        group_max = df_group.max().astype(np.float16)
        group_std = df_group.std().astype(np.float16)
        df[f'{cat_col}_min'] = df[f'{cat_col}'].map(group_min)
        df[f'{cat_col}_max'] = df[f'{cat_col}'].map(group_max)
        df[f'{cat_col}_std'] = df[f'{cat_col}'].map(group_std)
        df[f'{cat_col}_range'] = df[f'{cat_col}_max'] - df[f'{cat_col}_min']
        group_Q1 = df_group.quantile(0.25).astype(np.float16)
        group_Q2 = df_group.median().astype(np.float16)
        group_Q3 = df_group.quantile(0.75).astype(np.float16)
        df[f'{cat_col}_Q1'] = df[f'{cat_col}'].map(group_Q1)
        df[f'{cat_col}_Q2'] = df[f'{cat_col}'].map(group_Q2)
        df[f'{cat_col}_Q3'] = df[f'{cat_col}'].map(group_Q3)
        df[f'{cat_col}_IQR'] = df[f'{cat_col}_Q1'] - df[f'{cat_col}_Q3']
        del group_min, group_max, group_std, group_Q1, group_Q2, group_Q3; gc.collect()
        
    if print_option:
        print("Generated features: apply_target_encode")
        if smooth:
            print(f"'{cat_col}_smoothed_mean',")
        elif smooth == False:
            print(f"'{cat_col}_mean',")
        if statistic:
            print(f"'{cat_col}_min',")
            print(f"'{cat_col}_max',")
            print(f"'{cat_col}_std',")
            print(f"'{cat_col}_range',")
            print(f"'{cat_col}_Q1',")
            print(f"'{cat_col}_Q2',")
            print(f"'{cat_col}_Q3',")
            print(f"'{cat_col}_IQR',")
    del df_group, group_mean; gc.collect()

def t_df_pre(t_df, meta_user):
    t_df = t_df.merge(meta_user, on='uid', how='left')
    t_df.loc[t_df.premium_expired == False, 'premium_expired'] = 0
    t_df.loc[t_df.premium_expired == True, 'premium_expired'] = 1
    
    # label encode
    apply_label_encode(t_df, ['gender', 'family_structure', 'premium_type'])
    
    # target encode
    apply_target_encode(t_df, 'premium_expired', 'gender', smooth=False, print_option=False)
    apply_target_encode(t_df, 'premium_expired', 'birth_year', smooth=False, print_option=False)
    apply_target_encode(t_df, 'premium_expired', 'family_structure', smooth=False, print_option=False)
    
    # freq feature
    assert 'birth_year' in t_df
    apply_freq_encode(t_df, 'birth_year')
    apply_freq_encode(t_df, 'family_structure')
    apply_freq_encode(t_df, 'gender')
    # apply_freq_encode(t_df, 'premium_registered_at')
    apply_freq_encode(t_df, 'premium_type')
    # apply_freq_encode(t_df, 'registered_at')
    
    # cyclical date feature
    for col in ['registered', 'premium_registered']:
        f = lambda x: x.split(' ')[0]
        t_df[f'{col}_at_date'] = pd.to_datetime(t_df[f'{col}_at'].apply(f), format='%Y-%m-%d')
        t_df[f'{col}_at_day'] = t_df[f'{col}_at_date'].dt.day.astype(np.int8)
        apply_target_encode(t_df, 'premium_expired', f'{col}_at_day', smooth=False, print_option=False)
        t_df[f'{col}_at_month'] = t_df[f'{col}_at_date'].dt.month.astype(np.int8)
        apply_target_encode(t_df, 'premium_expired', f'{col}_at_month', smooth=False, print_option=False)
        t_df[f'{col}_at_year'] = t_df[f'{col}_at_date'].dt.year.astype(np.int8)
        apply_target_encode(t_df, 'premium_expired', f'{col}_at_year', smooth=False, print_option=False)
        apply_cyclical(t_df, f'{col}_at_day')
        apply_cyclical(t_df, f'{col}_at_month')
        apply_cyclical(t_df, f'{col}_at_year')
        del t_df[f'{col}_at_date']#, t_df[f'{col}_at_day'], t_df[f'{col}_at_month'], t_df[f'{col}_at_year']
            
        f = lambda x: x.split(' ')[1]
        t_df[f'{col}_at_hour'] = t_df[f'{col}_at'].apply(f).astype(np.int8)
        apply_cyclical(t_df, f'{col}_at_hour')
        del t_df[f'{col}_at']#, t_df[f'{col}_at_hour']
    del f; gc.collect()
    
    # more birth_year features
    # t_df.birth_year.fillna(1987.0, inplace=True)
    
    gc.collect()

def account_pre(t_df, meta_account):
    # uid account count, count discrepancy
    group_df = meta_account.groupby('uid')
    uid_account_count_min = group_df.count().min(axis=1)
    uid_account_count_max = group_df.count().max(axis=1)
    uid_account_count_dis = uid_account_count_max - uid_account_count_min
    t_df['uid_account_count_min'] = t_df['uid'].map(uid_account_count_min)
    t_df['uid_account_count_max'] = t_df['uid'].map(uid_account_count_max)
    t_df['uid_account_count_dis'] = t_df['uid'].map(uid_account_count_dis)
    
    # uid is_manual ratio
    uid_is_manual_sum = group_df.is_manual.sum()
    uid_is_manual_ratio = group_df.is_manual.sum() / group_df.is_manual.count()
    t_df['uid_is_manual_sum'] = t_df['uid'].map(uid_is_manual_sum)
    t_df['uid_is_manual_ratio'] = t_df['uid'].map(uid_is_manual_ratio)
    
    # uid unique service id count
    uid_service_cat_id = group_df.service_category_id.unique().apply(lambda x: len(x))
    t_df['uid_service_id_count'] = t_df['uid'].map(uid_service_cat_id)
    
    # uid service id when is_manual 1
    temp = meta_account[meta_account.is_manual == 1].groupby('uid').service_category_id.unique().apply(lambda x: len(x))
    t_df['uid_unique_service_id_when_manual'] = t_df['uid'].map(temp)
    temp = meta_account[meta_account.is_manual == 1].groupby('uid').service_category_id.unique().apply(lambda x: 13 in x).astype(np.int8)
    t_df['uid_13_in_unique_service_id_when_manual'] = t_df['uid'].map(temp)
    
    # uid service id when is_manual 0
    temp = meta_account[meta_account.is_manual == 0].groupby('uid').service_category_id.unique().apply(lambda x: len(x))
    t_df['uid_unique_service_id_when_not_manual'] = t_df['uid'].map(temp)
    temp = meta_account[meta_account.is_manual == 0].groupby('uid').service_category_id.apply(lambda x: stats.mode(x)[0]).astype(np.int8)
    t_df['uid_mode_service_id_when_not_manual'] = t_df['uid'].map(temp)
    
    # uid most frequent service id
    temp = group_df.service_category_id.apply(lambda x: stats.mode(x)[0]).astype(np.int8)
    t_df['uid_mode_service_id'] = t_df['uid'].map(temp)
    del temp
    
    # uid date features
    for col in ['created', 'first_succeeded']:
        #f = lambda x: x.split(' ')[1]
        #meta_account[f'{col}_at_hour'] = meta_account[f'{col}_at'].apply(f).astype(np.int8)
        #apply_cyclical(t_df, f'{col}_at_hour')
        
        f = lambda x: x.split(' ')[0]
        meta_account[f'{col}_at'] = pd.to_datetime(meta_account[f'{col}_at'].apply(f), format='%Y-%m-%d')
        meta_account[f'{col}_at_day'] = meta_account[f'{col}_at'].dt.day#.astype(np.int8)
        meta_account[f'{col}_at_month'] = meta_account[f'{col}_at'].dt.month#.astype(np.int8)
        meta_account[f'{col}_at_year'] = meta_account[f'{col}_at'].dt.year#.astype(np.int16)
        #apply_cyclical(meta_account, f'{col}_at_day')
        #apply_cyclical(meta_account, f'{col}_at_month')
        #apply_cyclical(meta_account, f'{col}_at_year')
        
        for shift in [-3, -2, -1, 1, 2, 3]:
            group_df = meta_account.groupby('uid')[f'{col}_at']
            meta_account[f'{col}_at_shift_{shift}'] = group_df.shift(shift)
            meta_account[f'{col}_at_dis_{shift}'] = meta_account[f'{col}_at'] - meta_account[f'{col}_at_shift_{shift}']
            
            group_df = meta_account.groupby('uid')[f'{col}_at_dis_{shift}']
            t_df[f'{col}_at_discrepancy_{shift}_min'] = t_df['uid'].map(group_df.min()).astype(np.int64)
        
        # created/first_suceeded date mode
        for date_type in ['day', 'month', 'year']:
            group_df = meta_account.groupby('uid')
            temp = group_df[f'{col}_at_{date_type}'].apply(lambda x: x.mode())
            t_df[f'uid_{col}_{date_type}_mode'] = t_df['uid'].map(temp)
    
            apply_target_encode(t_df, 'premium_expired', f'uid_{col}_{date_type}_mode')
            apply_freq_encode(t_df, f'uid_{col}_{date_type}_mode')
        
            temp = group_df[f'{col}_at_{date_type}'].mean()
            t_df[f'uid_{col}_{date_type}_mean'] = t_df['uid'].map(temp)
            temp = group_df[f'{col}_at_{date_type}'].max()
            t_df[f'uid_{col}_{date_type}_max'] = t_df['uid'].map(temp)
            temp = group_df[f'{col}_at_{date_type}'].min()
            t_df[f'uid_{col}_{date_type}_min'] = t_df['uid'].map(temp)
            temp = group_df[f'{col}_at_{date_type}'].std()
            t_df[f'uid_{col}_{date_type}_std'] = t_df['uid'].map(temp)
        
        temp = group_df[f'{col}_at'].max() - group_df[f'{col}_at'].min()
        t_df[f'uid_{col}_span'] = t_df['uid'].map(temp).astype(np.int64)
        apply_freq_encode(t_df, f'uid_{col}_span')
        apply_target_encode(t_df, 'premium_expired', f'uid_{col}_span')
    
    del temp; gc.collect()

def access_pre(t_df, meta_access):
    group_df = meta_access.groupby('uid')
    
    # label encode
    apply_label_encode(meta_access, ['channel_flag', 'url_type'])
    
    # uid channel flag mode
    temp = group_df.channel_flag.apply(lambda x: x.mode())
    t_df['uid_channel_flag_mode'] = t_df['uid'].map(temp)
    
    # uid url type unique
    temp = group_df.url_type.unique().apply(lambda x: len(x))
    t_df['uid_url_type_unique'] = t_df['uid'].map(temp)
    temp = group_df.url_type.apply(lambda x: stats.mode(x)[0][0])
    t_df['uid_url_type_mode'] = t_df['uid'].map(temp)
    
    # access count stats
    temp = group_df.count().max(axis=1)
    t_df['uid_access_count'] = t_df['uid'].map(temp)
    temp = group_df.access_count.sum().astype(np.int32)
    t_df['uid_real_access_count'] = t_df['uid'].map(temp)
    
    # freq encode and stat
    apply_freq_encode(meta_access, 'channel_flag')
    temp = meta_access.groupby('uid').channel_flag.max()
    t_df['uid_mode_channel_flag_ratio'] = t_df['uid'].map(temp)
    apply_freq_encode(meta_access, 'url_type')
    temp = meta_access.groupby('uid').url_type.max()
    t_df['uid_mode_url_type_ratio'] = t_df['uid'].map(temp)
    
    # access date stat
    f = lambda x: x.split(' ')[1]
    meta_access['accessed_at_hour'] = meta_access.accessed_at.apply(f).astype(np.int8)
    
    f = lambda x: x.split(' ')[0]
    meta_access['accessed_at'] = pd.to_datetime(meta_access['accessed_at'].apply(f), format='%Y-%m-%d')
    meta_access['accessed_day'] = meta_access['accessed_at'].dt.day.astype(np.int8)
    meta_access['accessed_month'] = meta_access['accessed_at'].dt.month.astype(np.int8)
    meta_access['accessed_year'] = meta_access['accessed_at'].dt.year.astype(np.int16)
    #apply_cyclical(meta_account, f'{col}_at_day')
    #apply_cyclical(meta_account, f'{col}_at_month')
    #apply_cyclical(meta_account, f'{col}_at_year')
    
    # date modes
    for date_type in ['day', 'month', 'year']:
        group_df = meta_access.groupby('uid')[f'accessed_{date_type}']
        temp = group_df.apply(lambda x: x.mode()[0])
        t_df[f'accessed_{date_type}_mode'] = t_df['uid'].map(temp)
        temp = group_df.mean()
        t_df[f'accessed_{date_type}_mean'] = t_df['uid'].map(temp)
        temp = group_df.max()
        t_df[f'accessed_{date_type}_max'] = t_df['uid'].map(temp)
        temp = group_df.min()
        t_df[f'accessed_{date_type}_min'] = t_df['uid'].map(temp)
    
    for shift in [-3, -2, -1, 1, 2, 3]:
        group_df = meta_access.groupby('uid')['accessed_at']
        meta_access[f'accessed_at_shift_{shift}'] = group_df.shift(shift)
        meta_access[f'accessed_at_dis_{shift}'] = meta_access['accessed_at'] - meta_access[f'accessed_at_shift_{shift}']
    
        group_df = meta_access.groupby('uid')[f'accessed_at_dis_{shift}']
        t_df[f'accessed_at_discrepancy_{shift}_min'] = t_df['uid'].map(group_df.min()).astype(np.int64)
    del meta_access['accessed_at']

train_df = pd.read_csv('train.csv')
reduce_mem_usage(train_df)
gc.collect()

test_foruser = pd.read_csv('test_foruser.csv')
reduce_mem_usage(test_foruser)
gc.collect()

meta_user = pd.read_csv('user.csv')
reduce_mem_usage(meta_user)
gc.collect()

meta_account = pd.read_csv('account.csv')
reduce_mem_usage(meta_account)
gc.collect()

meta_access = pd.read_csv('access.csv')
reduce_mem_usage(meta_access)
gc.collect()

#channel_flag = pd.read_csv('channel_flag.csv')
#service_category = pd.read_csv('service_category.csv')
#url_type = pd.read_csv('url_type.csv')

sample_submission = pd.read_csv('answer_sample.csv')

#prof = pdp.ProfileReport(meta_access)
#prof.to_file(outputfile="meta_access.html")

t_df_pre(train_df, meta_user)
gc.collect()
account_pre(train_df, meta_account)
gc.collect()
access_pre(train_df, meta_access)
gc.collect()

del meta_user, meta_account, meta_access

meta_user = pd.read_csv('user.csv')
reduce_mem_usage(meta_user)
gc.collect()
meta_account = pd.read_csv('account.csv')
reduce_mem_usage(meta_account)
gc.collect()
meta_access = pd.read_csv('access.csv')
reduce_mem_usage(meta_access)
gc.collect()

t_df_pre(test_foruser, meta_user)
gc.collect()
account_pre(test_foruser, meta_account)
gc.collect()
access_pre(test_foruser, meta_access)
gc.collect()

def lgb_train(train_set, valid_set, metric,
              lr, nl, md, bf, ff, mcw, mdil, l1, l2):
    import lightgbm as lgb
    params = {'objective': 'binary',
              'metric': metric,
              'boosting': 'gbdt',
              'seed': 8982,
              'learning_rate': lr,
              'num_leaves': int(nl),
              'max_depth': int(md),
              'bagging_freq': int(5),
              'bagging_fraction': bf,
              'feature_fraction': ff,
              'min_child_weight': mcw,   
              'min_data_in_leaf': int(mdil),
              'lambda_l1': l1,
              'lambda_l2': l2}

    # ===== train_set: expects tuple (X_train, y_train) =====
    X_train, y_train = train_set
    # ===== valid_set: expects tuple (X_valid, y_valid) =====
    X_valid, y_valid = valid_set

    # ===== define data in lgb terms =====
    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=category_cols)
    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=category_cols)
    watchlist = [d_train, d_valid]

    learning_curve = {}
    model = lgb.train(params,
                      train_set=d_train,
                      valid_sets=watchlist,
                      num_boost_round=600,
                      evals_result=learning_curve,
                      verbose_eval=200,
                      early_stopping_rounds=20)
    
    best_score = {f'train_{metric}': model.best_score['training'][f'{metric}'],
                  f'valid_{metric}': model.best_score['valid_1'][f'{metric}']}
    return model, best_score, learning_curve


def lgb_kfold(X_train, y_train,
              learning_rate=0.05, num_leaves=31, max_depth=-1,
              bagging_fraction=0.9, feature_fraction=0.9,
              min_child_weight=1e-3, min_data_in_leaf=20,
              lambda_l1=0.0, lambda_l2=0.0,
              bayes_opt=True):
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8982)
    metric = 'auc'
    #cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
    #print(cat_features)
    
    oofs = np.zeros(X_train.shape[0],)
    models = []; learning_curves = []; best_scores = []; valid_score = []
    for train_idx, valid_idx in kf.split(X_train, y_train):
        train_data = X_train.iloc[train_idx,:], y_train[train_idx]
        valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

        model, best_score, learning_curve = lgb_train(train_set=train_data,
                                                      valid_set=valid_data,
                                                      metric=metric,
                                                      lr=learning_rate,
                                                      nl=num_leaves,
                                                      md=max_depth,
                                                      bf=bagging_fraction,
                                                      ff=feature_fraction,
                                                      mcw=min_child_weight,
                                                      mdil=min_data_in_leaf,
                                                      l1=lambda_l1,
                                                      l2=lambda_l2)
        oofs[valid_idx] = model.predict(X_train.iloc[valid_idx,:], num_iteration=model.best_iteration)
        models.append(model)
        learning_curves.append(learning_curve)
        best_scores.append(best_score)
        gc.collect()
        valid_score.append(best_score[f'valid_{metric}'])
    valid_std_score = np.std(valid_score)
    valid_avg_score = np.mean(valid_score)
    
    
    if bayes_opt:
        return -valid_avg_score
    else:
        return valid_avg_score, valid_std_score, models, oofs#best_scores, learning_curves

def iterative_cv(X_train_full, y_train, added_cols):
    # ===== added_cols assumes list of evaluating feature names =====
    # ===== e.g. list(set(X_train_full.columns) - set(common_cols)) =====
    init_valid_avg_score = lgb_kfold(X_train_full[common_cols + category_cols], y_train, bayes_opt=True)
    init_valid_avg_score *= -1
    print(f'Current best score is {init_valid_avg_score}')
    options = []; discards = []
    for col in added_cols:
        
        X_train = X_train_full[common_cols + [col] + category_cols]
        new_valid_avg_score = lgb_kfold(X_train, y_train)
        new_valid_avg_score *= -1
        degree = new_valid_avg_score - init_valid_avg_score
        if degree > 0: # degree < 0 if objective is to minimize metric.
            options.append((col, degree))
            print(f"\nFeature '{col}', improved CV score by {degree}\n")
        else:
            discards.append((col, degree))
    options.sort(key=lambda tup: tup[1])
    discards.sort(key=lambda tup: tup[1])
    del col, X_train, degree; gc.collect()
    return options, discards

def bruteforce_combination(df, subject_cols, choose=2, print_option=True):
    from itertools import combinations
    comb = combinations(subject_cols, choose)
    for feat_1, feat_2 in comb:
        df[f'{feat_1}_.*_{feat_2}'] = df[f'{feat_1}'] * df[f'{feat_1}']
        df[f'{feat_1}_.-_{feat_2}'] = df[f'{feat_1}'] - df[f'{feat_1}']
        if print_option:
            #print('Generated features: bruteforce_feature_combination')
            print(f"'{feat_1}_.*_{feat_2}',")
            print(f"'{feat_1}_.-_{feat_2}',")
    del comb, feat_1, feat_2; gc.collect()

def plot_feature_importance(model):
    importance_df = pd.DataFrame(model.feature_importance(),
                                 index=[common_cols + category_cols],
                                 columns=['importance']).sort_values('importance')
    fig, ax = plt.subplots(figsize=(10, 13))
    importance_df.plot.barh(ax=ax)
    fig.show()

def bayes_opt_lgbm(init_points=20, n_iteration=80):
    from bayes_opt import BayesianOptimization
    bounds = {'learning_rate': (0.001, 0.3),
              'num_leaves': (20, 500), 
              'bagging_fraction' : (0.1, 1),
              'feature_fraction' : (0.1, 1),
              'min_child_weight': (0.001, 0.99),   
              'min_data_in_leaf': (3, 200),
              'max_depth': (-1, 100),
              'lambda_l1': (0.1, 300), 
              'lambda_l2': (0.1, 300)}
    optimizer = BayesianOptimization(f=lgb_kfold, pbounds=bounds, random_state=8982)
    optimizer.maximize(init_points=init_points, n_iter=n_iteration)
    
    print('Best score:', -optimizer.max['target'])
    print('Best set of parameters:')
    print(optimizer.max['params'])
    param = optimizer.max['params']; cv = -optimizer.max['target']
    return param, cv

def f1_threshold_search(y_true, y_proba, linspace=100):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(linspace)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

del_cols = ['uid', 'premium_expired']
category_cols = ['gender',
                 'birth_year',
                 'family_structure',
                 'premium_type']
common_cols = list(set(train_df.columns) - set(del_cols) - set(category_cols))

added_cols = []
"""
del_col = ['birth_year_IQR',
                 'birth_year_Q1',
                 'birth_year_Q2',
                 'birth_year_Q3',
                 'birth_year_max',
                 'birth_year_min',
                 'birth_year_range',
                 'birth_year_std',
                 'uid_succeed_span_mean',#
               'uid_account_span_mean',#
               'gender_mean',
               'birth_year_mean',
               'family_structure_mean']
comb_cols = list(set(common_cols) - set(del_cols))
bruteforce_combination(train_df, comb_cols, choose=2, print_option=False)
common_cols = list(set(train_df.columns) - set(del_cols))
"""
X_train_full = train_df[common_cols + category_cols]
X_test_full = test_foruser[common_cols + category_cols]

y_train = train_df['premium_expired'].values

#vas1, vss1, models1 = lgb_kfold(X_train_full_1, y_train_1, bayes_opt=False)
#vas2, vss2, models2 = lgb_kfold(X_train_full_2, y_train_2, bayes_opt=False)
vas, vss, models, oofs = lgb_kfold(X_train_full, y_train, bayes_opt=False)

#print()
#print(f'valid_avg_score: {vas1}')
#print(f'valid_std_score: {vss1}')
#print()
#print(f'valid_avg_score: {vas2}')
#print(f'valid_std_score: {vss2}')
#print()
print(f'valid_avg_score: {vas}')
print(f'valid_std_score: {vss}')
result = f1_threshold_search(y_train, oofs)
print(result)

#options, discards = iterative_cv(X_train_full, y_train, added_cols)

def pred(X_test, models):
    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'Predicting with {i}-th model')
        y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
        y_test_pred_total += y_pred_test
    y_test_pred_total /= len(models)
    return y_test_pred_total

y_test = pred(X_test_full, models)