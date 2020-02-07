import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import warnings; warnings.simplefilter('ignore')

"""def df_parallelize_run(df, func):
    import multiprocessing
    num_partitions, num_cores = psutil.cpu_count(), psutil.cpu_count()
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df"""

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

def category_concat(df, subject_cols):
    na_col = list(df.columns[df.isna().any()])
    for col in na_col:
        df[col].fillna('', inplace=True)
    temp_str = ''
    for col in subject_cols:
        temp_str += '_' + col
    df[temp_str[1:]] = ''
    for col in subject_cols:
        df[temp_str[1:]] += df[col]
    del na_col, temp_str, col; gc.collect()
        
def apply_row_nan(df):
    df['row_nan'] = df.isna().sum(axis=1).astype(np.int8)

def apply_cyclical(df, str_col):
    # df["hour"] = df["timestamp"].dt.hour
    # ===== assumes integer array =====
    df -= df[str_col].min()
    int_max = df[str_col].max()
    df[f'{str_col}_sin'] = np.sin(2 * np.pi * df[str_col] / int_max)
    df[f'{str_col}_cos'] = np.cos(2 * np.pi * df[str_col] / int_max)
    del int_max; gc.collect()
    
def apply_clip(df, str_col, pct_lower, pct_upper):
    LB, UB = np.percentile(df[str_col], [pct_lower, pct_upper])
    df[str_col] = np.clip(df[str_col], LB, UB)
    del LB, UB; gc.collect()

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

def apply_label_encode(df, subject_cols):
    for str_col in subject_cols:
        # ===== assumes Series of string =====
        temp_dict = {value: i for i, value in enumerate(df[str_col].unique())}
        df[str_col] = df[str_col].map(temp_dict)
    del temp_dict, str_col; gc.collect()

def apply_pca(df, subject_cols, r_after):
    # ===== assume input r_after is int =====
    # ===== assume subject_cols is list =====
    from sklearn.decomposition import PCA
    pca = PCA(n_components=r_after)
    r_prior = len(subject_cols)
    
    # ===== if vector space R2 to R1 =====
    if r_prior == 2:
        str_col_1, str_col_2 = subject_cols
        col = f'pc_{r_after}_{str_col_1[:6]}_{str_col_2[:6]}'
        df[col] = pca.fit_transform(df[subject_cols])
        print('Generated feature: apply_pca')
        print(f"'{col}',")
        del str_col_1, str_col_2; gc.collect()
        
    # ===== if vector space to Rr_after =====
    elif r_prior > 2:
        for i in range(r_after):
            col = f'pc_{i+1}_from_{r_prior}'
            df[col] = pca.fit_transform(df[subject_cols])
            print('Generated feature: apply_pca')
            print(f"'{col}',")
            del i; gc.collect()
    del pca, col, r_prior; gc.collect()

def apply_mov_stat(df, str_col, list_windows, shift=False, print_option=True):
    # ===== assume time is aligned =====
    for win in list_windows:
        rolled = df[str_col].rolling(window=win, min_periods=0)
        mov_avg = rolled.mean().reset_index() #.astype(np.float16)
        mov_max = rolled.max().reset_index() #.astype(np.float16)
        mov_min = rolled.min().reset_index() #.astype(np.float16)
        mov_std = rolled.std().reset_index() #.astype(np.float16)
        # mov_Q1 = rolled.quantile(0.25).reset_index() #.astype(np.float16)
        # mov_Q2 = rolled.quantile(0.5).reset_index() #.astype(np.float16)
        # mov_Q3 = rolled.quantile(0.75).reset_index() #.astype(np.float16)

        if shift:
            formula = int((win/2) - win)
            df[f'{str_col}_movavg_{win}'] = mov_avg[f'{str_col}'].shift(formula)
            df[f'{str_col}_movmax_{win}'] = mov_max[f'{str_col}'].shift(formula)
            df[f'{str_col}_movmin_{win}'] = mov_min[f'{str_col}'].shift(formula)
            df[f'{str_col}_movstd_{win}'] = mov_std[f'{str_col}'].shift(formula)
            # df[f'{str_col}_movQ1_{win}'] = mov_Q1[f'{str_col}'].shift(formula)
            # df[f'{str_col}_movQ2_{win}'] = mov_Q2[f'{str_col}'].shift(formula)
            # df[f'{str_col}_movQ3_{win}'] = mov_Q3[f'{str_col}'].shift(formula)
            del formula
        else:
            df[f'{str_col}_movavg_{win}'] = mov_avg[f'{str_col}']
            df[f'{str_col}_movmax_{win}'] = mov_max[f'{str_col}']
            df[f'{str_col}_movmin_{win}'] = mov_min[f'{str_col}']
            df[f'{str_col}_movstd_{win}'] = mov_std[f'{str_col}']
            # df[f'{str_col}_movQ1_{win}'] = mov_Q1[f'{str_col}']
            # df[f'{str_col}_movQ2_{win}'] = mov_Q2[f'{str_col}']
            # df[f'{str_col}_movQ3_{win}'] = mov_Q3[f'{str_col}']
        
        if print_option:
            print('Generated features: apply_mov_stat')
            print(f"'{str_col}_movavg_{win}',")
            print(f"'{str_col}_movmax_{win}',")
            print(f"'{str_col}_movmin_{win}',")
            print(f"'{str_col}_movstd_{win}',")
            #print(f"'{str_col}_movQ1_{win}',")
            #print(f"'{str_col}_movQ2_{win}',")
            #print(f"'{str_col}_movQ3_{win}',")
            
    del win, rolled, mov_avg, mov_max, mov_min, mov_std; gc.collect()
    # del mov_Q1, mov_Q2, mov_Q3; gc.collect()

def apply_interpolation(df, subject_cols, int_order, supp_median_fill=False):
    lin = lambda var: var.interpolate(method='linear', limit_direction='both')
    pol = lambda var: var.interpolate(method='polynomial', order=int_order, limit_direction='both')
    # ===== in ASHRAE, grouping was done via site_id =====
    # linear = df.groupby(grouping_col).apply(lin)
    # polyno = df.groupby(grouping_col).apply(pol)
    linear = df[subject_cols].apply(lin)
    polyno = df[subject_cols].apply(pol)
    df[subject_cols] = (linear[subject_cols] + polyno[subject_cols]) * 0.5
    
    # ===== if missing value remains: =====
    if supp_median_fill:
        #[col for col in cols if temp[col].isna().sum() > 0]
        for col in subject_cols:
            df[col].fillna(df[col].median(), inplace=True)
            del col
    del lin, pol, linear, polyno; gc.collect()


# input: df, subject_cols
# 初期値依存性に対応するiterative process
# 次にエルボー法
def apply_kmeans(df, subject_cols, plot=True):
    from sklearn.cluster import KMeans
    elbow = []
    for i in range(1,5):
        try:
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(df[subject_cols])
            elbow.append(kmeans.inertia_)
        except:
            continue
    
    if plot:
        plt.plot(range(1,5), elbow, marker='o')
        plt.xlabel('# of clusters')
        plt.ylabel('SSE')
        plt.show()
    # labels = kmeans.labels_
    # color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF', 3:'#04484C'}
    # colors = [color_codes[x] for x in labels]
    # plt.scatter(aiueo[:,0], aiueo[:,1], color=colors)
    # plt.show()
    # https://qiita.com/deaikei/items/11a10fde5bb47a2cf2c2
    del elbow, i; gc.collect()
    
def bruteforce_combination(df, subject_cols, choose=2, print_option=True):
    from itertools import combinations
    comb = combinations(subject_cols, choose)
    for feat_1, feat_2 in comb:
        df[f'{feat_1}_.*_{feat_2}'] = df[f'{feat_1}'] * df[f'{feat_1}']
        df[f'{feat_1}_.-_{feat_2}'] = df[f'{feat_1}'] - df[f'{feat_1}']
        if print_option:
            print('Generated features: bruteforce_feature_combination')
            print(f"'{feat_1}_.*_{feat_2}',")
            print(f"'{feat_1}_.-_{feat_2}',")
    del comb, feat_1, feat_2; gc.collect()

def apply_isna_feature(df, subject_cols):
    binary_isna = [col+"_isnan" for col in subject_cols]
    df[binary_isna] = df[subject_cols].isna().astype(int)
    del binary_isna; gc.collect()
    
def apply_oneth_feature(df, str_col):
    import math
    modify = np.vectorize(math.modf)
    oneth, tenth = modify(df[str_col] / 10)
    df[f'{str_col}_oneth'] = oneth * 10
    del tenth; gc.collect()
    
def apply_shift_feature(df, subject_cols, list_shift):
    for col in subject_cols:
        for step in list_shift:
            df[f'{col}_shift_{step}'] = df[col].shift(int(step))
    del col, step; gc.collect()

def apply_freq_encode(df, str_col):
    temp_dict = {sample: df.loc[df[str_col]==sample].shape[0] for sample in df[str_col].unique()}
    #for sample in df[str_col].unique():
    #    temp_dict[sample] = df.loc[df[str_col]==sample].shape[0]
    df[f'{str_col}_ratio'] = df[str_col].map(temp_dict) / df[str_col].shape[0]
    del temp_dict; gc.collect()

def apply_nonlinear(df, subject_cols):
    for col in subject_cols:
        temp_count = df[f'{col}'].isna().sum()
        df[f'{col}'] = np.log1p(df[f'{col}'])
        if df[f'{col}'].isna().sum() > temp_count:
            print(f"New nan in '{col}' via apply_nonlinear")
    del col, temp_count; gc.collect()

def create_X_y(train_df, target_meter):
    target_train_df = train_df[train_df['meter'] == target_meter]
    target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')
    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
    
    X_train = target_train_df#[common_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df; gc.collect()
    return X_train, y_train

def create_X(test_df, target_meter, train_df):
    target_test_df = test_df[test_df['meter'] == target_meter]
    target_test_df = target_test_df.merge(building_meta_df, on='building_id', how='left')
    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    
    X_test = target_test_df[common_cols + category_cols]
    return X_test

def plot_feature_importance(model):
    importance_df = pd.DataFrame(model.feature_importance(),
                                 index=common_cols + category_cols,
                                 columns=['importance']).sort_values('importance')
    fig, ax = plt.subplots(figsize=(10, 13))
    importance_df.plot.barh(ax=ax)
    fig.show()
    
