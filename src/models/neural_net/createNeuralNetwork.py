# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:25:42 2020

@author: javier.moral.hernan1
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K

class_weight_manual = {0: 200.,
                       1: 15.,
                       2: 40.,
                       3: 150.,
                       4: 30.,
                       5: 1.,
                       6: 40.}
filepath_="weights.best.hdf5"
best_epoch = ModelCheckpoint(filepath=filepath_, monitor='val_accuracy',
                             save_best_only=True)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
class NeuralNetModel():
    def __init__(self, data):
        self.data = data
        self.createModel()
        self.ES = EarlyStopping(monitor='val_acc',
                                min_delta=0,
                                patience=10,
                                verbose=0, 
                                mode='auto',
                                baseline=None)

    def createModel(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.data.shape[1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(400))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(250))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(150))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(7))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=[f1])
        self.model = model
    