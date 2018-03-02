import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from keras import backend as K
from helpers import format_submission

path_to_file = '~/Desktop/WiDS2018Datathon/'
num_classes = 2
seed = 37

print 'Loading data...\n'
x_train = pd.read_csv(path_to_file+'data/x_train.csv')
y_train = pd.read_csv(path_to_file+'data/y_train.csv')
#x_test = pd.read_csv(path_to_file+'data/x_test.csv')

# Split into train and test
trainX, testX, trainY, testY = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

input_shape = trainX.shape[1]
trainX = trainX.values.reshape(trainX.shape[0], trainX.shape[1])
trainY = trainY.values.reshape(trainY.shape[0],)
trainY = to_categorical(trainY, num_classes)
testX = testX.values.reshape(testX.shape[0], testX.shape[1])
testY = testY.values.reshape(testY.shape[0],)
testY = to_categorical(testY, num_classes)

print 'x_train shape: {}'.format(trainX.shape)
print 'y_train shape: {}'.format(trainY.shape)
print 'x_test shape: {}'.format(testX.shape)
print 'y_test shape: {}\n'.format(testY.shape)

print 'Defining model...\n'

units = 32
optimizer = 'Adagrad'
loss = 'binary_crossentropy'
activation = 'sigmoid'

model = Sequential()
model.add(Dense(units, activation='relu', input_shape=(input_shape,)))
model.add(Dense(units, activation='relu'))
model.add(Dense(units, activation='relu'))
model.add(Dense(num_classes, activation=activation)) 

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print 'Fitting model...\n'
model.fit(trainX, trainY, epochs=5, batch_size=32)

print '\nEvaluating model...\n'
print 'Number of units: {}'.format(units)
print 'Optimizer: {}'.format(optimizer)
print 'Loss: {}'.format(loss)
print 'Activation Fxn Final Layer: {}\n'.format(activation)

score_test = model.evaluate(testX, testY, verbose=0)
print 'Test loss: {}'.format(score_test[0])
print 'Test accuracy: {}\n'.format(score_test[1])

score_train = model.evaluate(trainX, trainY, verbose=0)
print 'Train loss: {}'.format(score_train[0])
print 'Train accuracy: {}\n'.format(score_train[1])

preds = model.predict_proba(testX)
score = roc_auc_score(testY, preds)
print 'Test ROC AUC: {}'.format(score)

# print 'Making predictions...\n'
# preds = model.predict_proba(x_test, batch_size=128)
# print preds

# print 'Formatting predictions DataFrame...\n'
# df = format_submission(preds, len(x_test))
# print df.head()

# print '\nExporting predictions to CSV...\n'
# sub_number = 5
# df.to_csv(path_to_file+'submissions/submission{}.csv'.format(sub_number), index=False)

print 'Program complete.'
