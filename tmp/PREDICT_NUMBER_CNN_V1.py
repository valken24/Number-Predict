#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras import Sequential


(x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()

BATCH_SIZE = 128
EPOCHS = 10
H = 28
W = 28
C = 1
INPUT_SIZE = (H,W,C)

x_train = x_train_original.reshape(x_train_original.shape[0], H, W, C)
x_test = x_test_original.reshape(x_test_original.shape[0], H, W, C)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train_original)
y_test = keras.utils.to_categorical(y_test_original)


def create_Model_v1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=INPUT_SIZE, activation='relu', name='Input_Conv2D_1'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', name='Conv2D_2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='MaxPool2D_1'))
    model.add(Dropout(0.5,name='Dropout_1'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', name='Conv2D_3'))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu', name='Conv2D_4'))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu', name='Conv2D_5'))
    model.add(MaxPooling2D(pool_size=(2,2), name='MaxPool2D_2'))
    model.add(Dropout(0.5,name='Dropout_2'))

    model.add(Flatten())
    model.add(Dense(500, activation='relu', name="Neural_1"))
    model.add(Dense(10, activation='softmax', name="Output"))

    return model

cnn_model_v1 = create_Model_v1()

cnn_model_v1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])

cnn_model_v1.summary()


cnn = cnn_model_v1.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test), shuffle=True)

evaluate = cnn_model_v1.evaluate(x_test, y_test)
score = evaluate[1] * 100
print("ACC: ", score, "%")


def test_predict(n):
    
    plt.imshow(x_test_original[n])
    plt.show()

    pred = x_test_original[n].reshape(1,28,28,1)  
    v = cnn_model_v1.predict(pred)
    print(v[0])

    for val in v:
        for value in range(len(val)):
            if val[value] == 1:
                print("Valor Predicho: ", value)


cnn_model_v1.save_weights("cnn_weights.h5")

cnn_model_v1.load_weights("cnn_weights.h5")


evaluate = cnn_model_v1.evaluate(x_test, y_test)
score = evaluate[1] * 100
print("ACC: ", score, "%")


def test_predict(n):
    
    plt.imshow(x_test_original[n])
    plt.show()

    pred = x_test_original[n].reshape(1,28,28,1)  
    v = cnn_model_v1.predict(pred)
    print(v[0])

    for val in v:
        for value in range(len(val)):
            if val[value] == 1:
                print("Valor Predicho: ", value)


test_predict(1000)



import mss
import time



def screen_record_efficient():
    mon = {"top": 823, "left": 1175, "width": 50, "height": 50}

    title = "[MSS] FPS benchmark"
    sct = mss.mss()

    while True:
        last_time = time.time()
        img = np.asarray(sct.grab(mon))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_resize = cv2.resize(gray,(28,28))
        
        cv2.imshow(title, gray_resize)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        print("fps: {}".format(np.around(1 / (time.time() - last_time))))
    sct.close()

print("MSS:", screen_record_efficient())


# In[ ]:




