from __future__ import division, print_function, absolute_import
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

batch_size = 128
nb_classes = 46
nb_epoch = 12

pickle_file = 'devanagari.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, 1, 32, 32)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(nb_classes) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

model = Sequential()

model = Sequential()
model.add(Convolution2D(100, 5, 5, border_mode='valid', input_shape=(1, 32, 32)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))
model.add(Convolution2D(250, 5, 5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(46))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_dataset, train_labels, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(valid_dataset, valid_labels))
score = model.evaluate(test_dataset, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
