# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:12:01 2020

@author: rucha
"""
# using CIFAR dataset, in which 60000 images in training dataset and 10000 in test dataset
import tensorflow as tf
import os
import numpy as np
import sys
from matplotlib import pyplot as plt

if not os.path.isdir('models'):
    os.mkdir('models')
print('Tensorflow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())

def get_three_classes(x,y): #this function is returning the smaller dataset we intend to use
#using a subset of the original dataset using 3 classes with 15000 images in training dataset
    indices_0,_=np.where(y==0.) #returns all those indices where label y=0
    indices_1,_=np.where(y==1.)
    indices_2,_=np.where(y==2.)
    indices=np.concatenate([indices_0, indices_1, indices_2], axis=0)
    x=x[indices] #only select the needed indices
    y=y[indices]
    count=x.shape[0] #counting the total number of examples
    indices=np.random.choice(range(count), count, replace=False) #shuffle the indices replace=False because we dont want any values to be repeated
    x=x[indices]
    y=y[indices]
    y=tf.keras.utils.to_categorical(y)
    return x,y

(x_train, y_train), (x_test, y_test)= tf.keras.datasets.cifar10.load_data()
x_train, y_train=get_three_classes(x_train, y_train)
x_test, y_test=get_three_classes(x_test, y_test)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

class_names=['aeroplane', 'car', 'bird']

def show_random_examples(x,y,p):
    indices=np.random.choice(range(x.shape[0]), 10, replace=False)
    x=x[indices]
    y=y[indices]
    p=p[indices]
    plt.figure(figsize=(10,15))
    for i in range(10):
        plt.subplot(5,2,1+i)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col='green' if np.argmax(y[i])==np.argmax(p[i]) else 'red'
        plt.xlabel(class_names[np.argmax(p[i])], color=col)
    plt.show()    

show_random_examples(x_train, y_train, y_train)
show_random_examples(x_test, y_test, y_test)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Input, Dense

def create_model():
    def add_conv_block(model, num_filters):
        model.add(Conv2D(num_filters,3, activation='relu', padding='same'))
        model.add(BatchNormalization()) #ensures that there is not much covariance shift in the preceeding layer
        model.add(Conv2D(num_filters,3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2)) #reduces the size to half
        model.add(Dropout(0.5))
        return model
    model= tf.keras.models.Sequential()
    model.add(Input(shape=(32,32,3)))
    model=add_conv_block(model, 32)
    model=add_conv_block(model, 64)
    model=add_conv_block(model, 128)
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model=create_model()
model.summary()

h=model.fit(x_train/255., y_train, validation_data=(x_test/255., y_test), epochs=10, batch_size=128, 
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
                       tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5', save_best_only=True,
                                                          save_weights_only=False, monitor='val_accuracy')])
accs=h.history['accuracy']
val_accs=h.history['val_accuracy']
plt.plot(range(len(accs)), accs, label='Training')
plt.plot(range(len(val_accs)), val_accs, label='Validation')
plt.legend()
plt.show()
preds=model.predict(x_test/255.)
show_random_examples(x_test, y_test, preds)