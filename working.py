import pandas as pd
from pandas.core.reshape import  get_dummies
from collections import defaultdict
from sklearn import cross_validation
from sklearn.metrics import hamming_loss, log_loss, roc_auc_score
import time
import numpy as np
from model_info import get_pipeline, get_rf_pipeline
from load_data import get_train_test

import argparse


def remove_train_data(train, test):
    cat_columns = [m for m in train.columns if 'c_' in m]
    for col in cat_columns:
        poss_vals = pd.unique(train[col])
        cur_values = pd.unique(test[col])
        unused = [entry for entry in poss_vals if entry not in cur_values ]
        for entry in unused:
            train = train[train[col] != entry]
    return train

def validate_model_holdout(train):

    results = dict()
    models = dict()
    preds = []
    grouped = train.groupby('release')

    split_data_train = pd.concat([df[0:int(df.shape[0] * .80)] for name, df in grouped], axis=0)
    split_data_train = split_data_train.fillna(0)

    split_data_holdout = pd.concat([df[int(df.shape[0] * .20):] for name, df in grouped], axis=0)
    split_data_holdout = split_data_holdout.fillna(0)

    for output in outputs:
        print "fit " + output
        pipe_clf = get_pipeline()
        pipe_clf.fit(split_data_train.drop(outputs, axis=1), split_data_train[output])
        preds = pipe_clf.predict_proba(split_data_holdout.drop(outputs, axis=1))
        X_test[output + 'pred'] = preds[0:, 1:]
        X[output + 'pred'] = X[output]
        results[output].append(log_loss(test_test, preds[0:, 1:]))
        print results[output]

    return models


def validate_model_sep(train, labels):
    results = defaultdict(list)
    sum_mean = 0
    sum_std = 0
    print train.shape
    df = train[0:int(train.shape[0] * .80)]
    labels = labels[0:int(train.shape[0] * .80)]
    print df.shape

    for output in labels.columns:
        cv = cross_validation.StratifiedKFold(labels[output])
        print outputs
        print output
        print "------------------------"
        for traincv, testcv in cv:
            train_sample = df.reset_index().loc[traincv]
            train_test = labels.reset_index()[output].loc[traincv]

            test_sample = df.reset_index().loc[testcv]
            test_test = labels.reset_index()[output].loc[testcv]

            t0 = time.time()

            pipe_clf = get_pipeline()
            pipe_clf.fit(train_sample, train_test)
            
            preds = pipe_clf.predict_proba(test_sample)
            results[output].append(log_loss(test_test, preds[0:, 1:]))
            print results[output]

            print "elapsed "
            print time.time() - t0

    for result in results:
        sum_mean += np.array(results[result]).mean()
        sum_std += np.array(results[result]).std()
        print "Results %s: " % (result,) + str( np.array(results[result]).mean()) + " " + str(np.array(results[result]).std())
    print "Final: %s %s" %(1.0 * sum_mean / len(results), 1.0 * sum_std / len(results))

X, X_test, X_labels = get_train_test()
outputs = X_labels.drop('id', axis=1).columns

print "loaded pickle"
#models = validate_model_holdout(X)
#validate_model_sep(X.drop(outputs, axis=1), X[outputs])
submission = pd.read_csv('submission.csv')

to_drop = []
for output in outputs:
    print "fit " + output
    pipe_clf = get_pipeline()
    pipe_clf.fit(X.drop(outputs, axis=1), X[output])
    preds = pipe_clf.predict_proba(X_test)
    X_test[output + 'pred'] = preds[0:, 1:]
    X[output + 'pred'] = X[output]
    to_drop.append(output + 'pred')
    submission[output] = preds[0:, 1:]
    
#for output in reversed(outputs):
#    print "fit " + output
#    pipe_clf = get_pipeline()
#    pipe_clf.fit(X.drop(outputs, axis=1).drop(to_drop, axis=1), X[output])
#    preds = pipe_clf.predict_proba(X_test.drop(to_drop, axis=1))
#    X_test[output + 'predr'] = preds[0:, 1:]
#    X[output + 'predr'] = X[output]
#    submission[output] = (submission[output].values + preds[0:, 1:]) / 2.0

submission.to_csv('resultsMixed.csv', index=False)
