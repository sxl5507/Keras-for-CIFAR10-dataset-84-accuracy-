# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:29:48 2018

@author: Siyan
"""
# CPU: i7 6700K, 4 GHz
# GPU: GTX 960, 6 GB
# 20s for each Epoch when batch_size= 300
# final accuracy around 84%, 15 epoch reaches 80%

from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Sequential # basic class for specifying and training a neural network
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
import numpy as np




batch_size = 300 # training examples in each iteration
num_epochs = 50 # iterate over the entire training set
kernel_size = 3
pool_size = 3
pool_size_reduce=1
conv_depth_32 = 32 
conv_depth_64 = 64
conv_depth_10 = 10
drop_prob_1 = 0.20 
drop_prob_2 = 0.5 
hidden_size = 512 


(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data
num_train, height, width, depth = X_train.shape # 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # 10 image classes


Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode 
Y_test = np_utils.to_categorical(y_test, num_classes)




# 3 × 3 conv (2 layers), 32 ReLU -> MaxPool -> BatchNormalization -> Dropout
model = Sequential()

model.add(Convolution2D(
        filters= conv_depth_32,
        kernel_size= kernel_size,
        padding= 'same',
        activation= 'relu',
        input_shape= (height, width, depth)))
model.add(BatchNormalization())
model.add(Convolution2D(
        filters= conv_depth_32,
        kernel_size= kernel_size,
        padding= 'same',
        activation= 'relu', ))
model.add(MaxPooling2D(pool_size=pool_size, strides=2)) # pooling before normalization
model.add(BatchNormalization())
model.add(Dropout(drop_prob_1))




# 3 × 3 conv (2 layers), 64 ReLU -> BatchNormalization -> MaxPool -> Dropout
model.add(Convolution2D(
        filters= conv_depth_64,
        kernel_size= kernel_size,
        padding= 'same',
        activation= 'relu'))
model.add(BatchNormalization())
model.add(Convolution2D(
        filters= conv_depth_64,
        kernel_size= kernel_size,
        padding= 'same',
        activation= 'relu'))
model.add(BatchNormalization()) # normalization before pooling  
model.add(MaxPooling2D(pool_size=pool_size, strides=2))
model.add(Dropout(drop_prob_1))




# 3 × 3 conv, 32 ReLU -> BatchNormalization
model.add(Convolution2D(
        filters= conv_depth_32,
        kernel_size= kernel_size,
        padding= 'same',
        activation= 'relu'))
model.add(BatchNormalization())



model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop_prob_2))
model.add(Dense(num_classes, activation='softmax'))





board= TensorBoard(log_dir=r'.\Tgraph', histogram_freq=0, write_graph=True, write_images=True)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_split=0.1,
          shuffle=True,
          callbacks=[board])
model.evaluate(X_test, Y_test, verbose=1)




