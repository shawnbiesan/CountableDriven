__author__ = 'sbiesan'
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import ProbabilisticPCA, RandomizedPCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

def get_pipeline():
    pipe_clf = Pipeline([
        ('pca', RandomizedPCA(n_components=500)),
        #('rf', RandomForestClassifier(n_jobs=-1, n_estimators=100))
        #('sgd', SGDClassifier(alpha=.000001, loss='log'))
        ('log', LogisticRegression(C=1000))
        
    ])
    print pipe_clf
    return pipe_clf
	
def get_rf_pipeline():
    pipe_clf = Pipeline([
        ('pca', RandomizedPCA(n_components=50)),
        ('rf', RandomForestClassifier(n_jobs=-1, n_estimators=200))
        #('sgd', SGDClassifier(alpha=.1))
        #('log', LogisticRegression(C=1000))
        
    ])
    print pipe_clf
    return pipe_clf