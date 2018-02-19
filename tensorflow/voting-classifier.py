import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from helpers import format_submission

seed = 37
num_classes = 2

print 'Loading data...\n'
path_to_file = '~/Desktop/WiDS2018Datathon/'
train = pd.read_csv(path_to_file+'data/train.csv', low_memory=False)
test = pd.read_csv(path_to_file+'data/test.csv', low_memory=False)

print 'Train Shape: {}'.format(train.shape)
print 'Test Shape: {}\n'.format(test.shape)

label = train['is_female']
del train['is_female']
del train['train_id']
del test['test_id']

# Remove rows/columns that are missing all data
train = train.dropna(axis=0, how='all')
train = train.dropna(axis=1, how='all')

# Convert to dummy variables
print 'Converting to dummy variables...\n'
train_str = train.applymap(str)
train_dummies = pd.get_dummies(train_str)

# Split into train and validation set
print 'Splitting into train and validation set...\n'
X_train, X_test, y_train, y_test = train_test_split(train_dummies, label, test_size=0.2, random_state=seed)

# # Format train and test set 
# print 'Converting to dummy variables...\n'
# train_str = train.applymap(str)
# test_str = test.applymap(str)
# total = pd.concat([train_str, test_str], ignore_index=True)
# total_dummies = pd.get_dummies(total)
# X_train = total_dummies.head(len(train))
# X_test = total_dummies.tail(len(test))

print 'Data preprocessing complete.\n'

# Feature selection with a Variance Threshold of 0
# print X_train.shape[1] # 8918
t=0
sel = VarianceThreshold(threshold=t)
X_train_new = pd.DataFrame(sel.fit_transform(X_train))
X_test_new = pd.DataFrame(sel.transform(X_test))

input_shape = X_train_new.shape[1] # 8636
# X_train_new = X_train_new.values.reshape(X_train_new.shape[0], X_train_new.shape[1])
y_train = y_train.values.reshape(y_train.shape[0],)
y_train = to_categorical(y_train, num_classes)
# X_test_new  = X_test_new.values.reshape(X_test_new.shape[0], X_test_new.shape[1])
y_test = y_test.values.reshape(y_test.shape[0],)
y_test = to_categorical(y_test, num_classes)

# Define neural network
def neural_network():
	units = 33
	model = Sequential()
	model.add(Dense(units, activation='relu', input_shape=(8636,)))
	model.add(Dense(units, activation='relu'))
	model.add(Dense(units, activation='relu'))
	model.add(Dense(num_classes, activation='softmax')) 
	model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
	return model

# Voting classifier
print 'Fitting model...\n'
# logreg = LogisticRegression(C=0.2, penalty='l1', solver='liblinear')
# xgb = XGBClassifier(max_depth=7, n_estimators=100, random_state=seed)
clf = KerasClassifier(build_fn=neural_network, epochs=5, batch_size=32)
# clf = VotingClassifier(estimators=[('logreg', logreg), ('xgb', xgb), ('net', net)], voting='soft')
clf = clf.fit(X_train_new.values, y_train)

print 'Making predictions...\n'
predictions = clf.predict_proba(X_test_new.values)
preds = np.round([p[1] for p in preds], 1)
score = roc_auc_score(y_test, preds)
print 'ROC AUC Score: {}\n'.format(score)

print 'Modeling complete.'
