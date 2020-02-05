import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import warnings; warnings.simplefilter('ignore')

# ===== How to train =====
# 1. Define datas: X_train_full, X_train, y_train
# 2. Define category_cols and common_cols
# 3. Normal training:
#    valid_avg_score, models,
#    best_scores, learning_curves = lgb_kfold(X_train, y_train,
#                                            parameters, bayes_opt=False)
# 4. Iterative validation:
#    Define added_cols
#    options, discards = iterative_cv(X_train_full, y_train, added_cols)
# 5. Bayes Opt:
#    param, cv = bayes_opt()
#    Make sure data is defined as X_train and y_train.
#    Also, make sure to comment out two inputs
#    X_train and y_train in lgb_kfold.

def lgb_train(train_set, valid_set, metric,
              lr, nl, md, bf, ff, mcw, mdil, l1, l2):
    import lightgbm as lgb
    params = {'objective': 'regression',
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
              learning_rate=0.1, num_leaves=31, max_depth=-1,
              bagging_fraction=0.9, feature_fraction=0.9,
              min_child_weight=1e-3, min_data_in_leaf=20,
              lambda_l1=0.0, lambda_l2=0.0,
              bayes_opt=True):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=8982)
    metric = 'rmse'
    #cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
    #print(cat_features)
    
    models = []; learning_curves = []; best_scores = []; valid_avg_score = 0
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
        models.append(model)
        learning_curves.append(learning_curve)
        best_scores.append(best_score)
        gc.collect()
        
        valid_avg_score += best_score[f'valid_{metric}']
    valid_avg_score /= len(models)
    
    if bayes_opt:
        return -valid_avg_score
    else:
        return valid_avg_score, models, best_scores, learning_curves

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
        if degree < 0: # degree < 0 if objective is to minimize metric.
            options.append((col, degree))
            print(f"\nFeature '{col}', improved CV score by {degree}\n")
        else:
            discards.append((col, degree))
    options.sort(key=lambda tup: tup[1])
    discards.sort(key=lambda tup: tup[1])
    del col, X_train, degree; gc.collect()
    return options, discards

def pred(X_test, models):
    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'Predicting with {i}-th model')
        y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
        y_test_pred_total += y_pred_test
    y_test_pred_total /= len(models)
    return y_test_pred_total

"""
sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
reduce_mem_usage(sample_submission)
sample_submission.loc[test_df['meter'] == 0, 'meter_reading'] = y_test0
sample_submission.loc[test_df['meter'] == 1, 'meter_reading'] = y_test1
sample_submission.loc[test_df['meter'] == 2, 'meter_reading'] = y_test2
sample_submission.loc[test_df['meter'] == 3, 'meter_reading'] = y_test3
#test_df.loc[(test_df.meter == 0) & (test_df.building_id.isin(malign_building_site0)), 'meter_reading'] *= 3.4118
#sample_submission = test_df[['row_id', 'meter_reading']]
sample_submission['meter_reading'] = sample_submission['meter_reading'].clip(lower=0)
sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
"""