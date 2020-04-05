# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 11:33:55 2020

@author: javier.moral.hernan1
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation


class AutoEncoderModel():
    def __init__(self, data):
        self.data = data
        self.input_dim = self.data.shape[1]
        self.createModel()

    def createModel(self):
        autoencoder = Sequential()

        # Encoder Layers
        autoencoder.add(Dense(50,
                              input_shape=(self.input_dim,),
                              activation='relu'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Dense(47, activation='relu'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Dense(43, activation='relu'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Dense(40, activation='relu'))
        autoencoder.add(BatchNormalization())
        
        # Decoder Layers
        autoencoder.add(Dense(43, activation='relu'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Dense(47, activation='relu'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Dense(50, activation='relu'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Dense(self.input_dim, activation='softmax'))

        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
        autoencoder.summary()
        self.model = autoencoder
        
        
# %% Autoencoder
# autoencoder = AutoEncoderModel(X_train)
# autoencoder.model.fit(X_train, X_train,
#                       epochs=250,
#                       batch_size=800,
#                       shuffle=True,
#                       validation_split=0.30,
#                       verbose=2)