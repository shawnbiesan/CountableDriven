__author__ = 'sbiesan'
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import ProbabilisticPCA, RandomizedPCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

def get_pipeline():
    pipe_clf = Pipeline([
        ('pca', RandomizedPCA(n_components=100)),
        #('rf', RandomForestClassifier(n_jobs=-1, n_estimators=30))
        #('sgd', SGDClassifier(alpha=.1))
        ('log', LogisticRegression(C=.1))
    ])

    return pipe_clf