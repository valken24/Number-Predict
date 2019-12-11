#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras import Sequential
from keras.datasets import mnist


# In[9]:


BATCH_SIZE = 128
EPOCHS = 20
H,W,C = 28,28,1

(x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()

x_train = x_train_original.reshape(x_train_original.shape[0], H, W, C)
x_test = x_test_original.reshape(x_test_original.shape[0], H, W, C)
INPUT_SIZE = (H,W,C)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train_original)
y_test = keras.utils.to_categorical(y_test_original)


# In[10]:


def create_Model_v1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=INPUT_SIZE, activation='relu', name='Input_Conv2D_1'))
    model.add(MaxPooling2D(pool_size=(2,2), name='MaxPool2D_1'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', name='Conv2D_2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='MaxPool2D_2'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name="Neural_1"))
    model.add(Dropout(0.25,name='Dropout_1'))
    model.add(Dense(128, activation='relu', name="Neural_2"))  
    model.add(Dropout(0.3,name='Dropout_2'))
    model.add(Dense(10, activation='softmax', name="Output"))

    return model


# In[11]:


cnn_model_v1 = create_Model_v1()
cnn_model_v1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])


# In[12]:


cnn_model_v1.summary()


# ## FIT MODEL AND EVALUATE

# In[13]:


cnn = cnn_model_v1.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test), 
                       shuffle=True)


# In[14]:


evaluate = cnn_model_v1.evaluate(x_test, y_test)
score = evaluate[1] * 100
print("ACC: ", score, "%")


# In[19]:


cnn_model_v1.save_weights("Weights H5/cnn_weights_v2.h5")


# ## LOAD PRETRAINED MODEL

# In[ ]:


cnn_model_v1.load_weights("Weights H5/cnn_weights.h5")


# In[ ]:


evaluate = cnn_model_v1.evaluate(x_test, y_test)
score = evaluate[1] * 100
print("ACC: ", score, "%")


# In[25]:


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
test_predict(505)


# In[26]:


import mss
import time


# In[ ]:


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

