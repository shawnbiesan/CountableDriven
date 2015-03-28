__author__ = 'sbiesan'
import pandas as pd
import numpy as np
import pickle
from col_info import num_columns, cat_columns

from pandas import get_dummies
import os.path


def get_train_test():

    X_labels = pd.read_csv('train_labels.csv')

    if(os.path.isfile('X_pickle.p') and os.path.isfile('X_pickle.p')):
        X = pickle.load(open('X_pickle.p', 'rb'))
        X_test = pickle.load(open('X_test_pickle.p', 'rb'))
        print 'loaded pickle'
    else:
        X = pd.concat([pd.read_csv('train.csv'), X_labels.drop('id', axis=1)], axis=1)
        X_test = pd.read_csv('test.csv')

        #cat_columns = [m for m in X.columns if 'c_' in m or m == 'release']

        print "filling"
        
        X[num_columns] = X[num_columns].fillna(0)
        print "1"
        X[cat_columns] = X[cat_columns].fillna(-1)
        print "2"
        X_test[num_columns] = X_test[num_columns].fillna(0)
        print "3"
        X_test[cat_columns] = X_test[cat_columns].fillna(-1)
        print "4"
        
        print "filled"
                
        for col in cat_columns:
            if len(pd.unique(X[col])) != len(pd.unique(X_test[col])):
                X.drop(col, axis=1)
                X_test.drop(col, axis=1)
            else:
                dummies = get_dummies(X[col], prefix=col)
                dummies_test = get_dummies(X_test[col], prefix=col)
                X[dummies.columns] = dummies
                X_test[dummies_test.columns] = dummies_test
        X = X.drop(cat_columns, axis=1).fillna(0)
        X_test = X_test.drop(cat_columns, axis=1).fillna(0)
        print "cleaned"

        pickle.dump(X, open('X_pickle.p', 'wb'))
        pickle.dump(X_test, open('X_test_pickle.p', 'wb'))
        print 'dumped pickle'
    return X, X_test, X_labels