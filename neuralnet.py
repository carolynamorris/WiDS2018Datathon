import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from helpers import format_submission

path_to_file = '~/Desktop/WiDS2018Datathon/'
num_classes = 2

print 'Loading data...\n'
x_train = pd.read_csv(path_to_file+'data/x_train.csv')
y_train = pd.read_csv(path_to_file+'data/y_train.csv')
x_test = pd.read_csv(path_to_file+'data/x_test.csv')

x_train = x_train.values.reshape(18255, 10348)
y_train = y_train.values.reshape(18255,)
y_train = to_categorical(y_train, num_classes)
x_test = x_test.values.reshape(27285, 10348)

print 'x_train shape: {}'.format(x_train.shape)
print 'y_train shape: {}'.format(y_train.shape)
print 'x_test shape: {}\n'.format(x_test.shape)

print 'Defining model...\n'
units = 57
model = Sequential()
model.add(Dense(units, activation='relu', input_shape=(10348,)))
model.add(Dense(num_classes, activation='softmax')) 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print 'Fitting model...\n'
model.fit(x_train, y_train, epochs=5, batch_size=32)

print '\nEvaluating model...\n'
score = model.evaluate(x_train, y_train, verbose=0)
print 'Train loss: {}'.format(score[0])
print 'Train accuracy: {}\n'.format(score[1])

print 'Making predictions...\n'
preds = model.predict_proba(x_test, batch_size=128)
print preds

print 'Formatting predictions DataFrame...\n'
df = format_submission(preds)
print df.head()

print '\nExporting predictions to CSV...\n'
sub_number = 6
df.to_csv(path_to_file+'submissions/submission{}.csv'.format(sub_number), index=False)

print 'Program complete.'
