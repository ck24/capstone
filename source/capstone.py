import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

random_state = 42
df = pd.read_csv('../data/train.csv')

target = df['ACTION']
features = df.drop('ACTION', axis=1)

print "Features shape: %s %s" % features.shape

v = DictVectorizer(sparse=False)
enc = OneHotEncoder()
vectorizedFeatures = v.fit_transform(features.to_dict(orient='records'))
X = enc.fit_transform(vectorizedFeatures).toarray()
print 'Vectorized features shape: %s %s' % X.shape

features_train, features_test, target_train, target_test = train_test_split(
    X, target, test_size=0.25, random_state=random_state)

# dumb model
mean = target_test.mean()
print 'Mean: %s' % mean
# examine class distribution of target 
print target_test.value_counts()

# confusion matrix
print(confusion_matrix(target_test, np.ones(target_test.shape[0])))

# logistic regression
#lrModel = LogisticRegression(random_state=random_state)
#lrModel.fit(features_train, target_train)
#predict = lrModel.predict(features_test)
#predict_proba = lrModel.predict_proba(features_test)[:, 1]
#print 'Accuracy score: %s' % accuracy_score(target_test, predict)
#print 'Area under ROC: %s' % roc_auc_score(target_test, predict_proba)
