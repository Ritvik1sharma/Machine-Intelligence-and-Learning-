#!/usr/bin/env python
# coding: utf-8

# In[28]:


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras import datasets
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.keras import regularizers
from tensorflow.python import keras as abc


def Neural_Net(lr, num_layers, layer_sizes, batch_size, ep, lamda, drop_out, x_train, y_train, act, x_test, y_test):
    x_train = x_train/255
    x_test = x_test/255
    
    y_train = abc.utils.to_categorical(y_train)
    y_test = abc.utils.to_categorical(y_test)
    print(x_train.shape)
    print(y_train.shape)
    model = abc.models.Sequential()
    for i in range(num_layers-1):
        if i==0:
            model.add(Dense(layer_sizes[1], input_shape= (784,), activation = act, use_bias = True, kernel_regularizer = regularizers.l2(lamda), bias_regularizer = regularizers.l2(lamda)))
            model.add(Dropout(drop_out))
        elif i!=(num_layers-2):
            model.add(Dense(layer_sizes[i+1], activation = act, use_bias = True, kernel_regularizer = regularizers.l2(lamda), bias_regularizer = regularizers.l2(lamda)))
            model.add(Dropout(drop_out))
        elif i==(num_layers-2):
            model.add(Dense(layer_sizes[i+1], activation = 'softmax', use_bias = True, kernel_regularizer = regularizers.l2(lamda), bias_regularizer = regularizers.l2(lamda)))
            
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs = ep, batch_size = batch_size)
    #calculating accuracy and loss on the validation data
    scores = model.evaluate(x = x_test, y =y_test, batch_size = batch_size, verbose = 1)
    print(scores)
    return scores

def cross_validate(k):
    data = pd.read_csv('2014EE10421.csv', header = None)
    x_vals = data.values[:, :784]
    y_vals = data.values[:, 784]
    
    subarray_len = len(x_vals)//k
    #25 epochs was regularization of 1.000000e-14 with a learning rate of 5.000000e-01 with 4 layers in a batch_size of 8 The validation loss of the model came out to be 0.331223, with a validation accuracy of 0.930667
    lambdas = [1e-14]#[0,1e-18, 1e-10,1e-7,1e-4,1e-1,1e2]
    l_rates = [0.5]#[1e-5,1e-3,1e-1,1e1,1e3]
    batch_size = [8]#[16,32,256]
    num_layers = [4]#[3, 4, 5]
    least_loss = float('inf')
    best_model = None
    ep = 25
    dp = 0.25
    # a dictionary of all models with hyperparameters as keys to the dictionary and the validation_loss as the data value 
    models = {}
    val_losses = []
    for n in num_layers:
        for lr in l_rates:
            for bs in batch_size:
                for lb in lambdas:
                    net_loss = 0
                    if n == 3:
                        layer_sizes = [784,250,10]
                        for j in range(k):
                            if(j!= k-1):
                                train_array = np.concatenate((x_vals[int(0*subarray_len):int(j*subarray_len)],x_vals[int((j+1)*subarray_len):]))
                                val_array = x_vals[int(j*subarray_len):int((j+1)*subarray_len)]
                                train_labels = np.concatenate((y_vals[int(0*subarray_len):int(j*subarray_len)],y_vals[int((j+1)*subarray_len):]))
                                val_labels =  y_vals[int(j*subarray_len):int((j+1)*subarray_len)]
                            else:
                                train_array = x_vals[0:(k-1)*subarray_len]
                                val_array = x_vals[(k-1)*subarray_len:]
                                train_labels = y_vals[0:(k-1)*subarray_len]
                                val_labels = y_vals[(k-1)*subarray_len:]
                   # (lr, num_layers, layer_sizes, batch_size, ep, lamda, drop_out, x_train, y_train, act, x_test, y_test)
                            scores = Neural_Net(lr,3,layer_sizes,bs,ep,lb, dp, train_array, train_labels,'relu', val_array,val_labels)
                            net_loss += scores[0]   #scores[0] contains the loss
                        #storing the model performance details in the dictionary
                        models[(lr,3,bs,ep,lb)] = net_loss 
                        val_losses.append(net_loss)
                        if (net_loss<least_loss):
                            best_lr = lr
                            best_bs =bs
                            best_num_layers=3
                            least_loss = net_loss
                        
                    if n == 4:
                        layer_sizes = [784,500,250,10]
                        for j in range(k):
                            if(j!= k-1):
                                train_array = np.concatenate((x_vals[int(0*subarray_length):int(j*subarray_length)],x_vals[int((j+1)*subarray_length):]))
                                val_array = x_vals[int(j*subarray_length):int((j+1)*subarray_length)]
                                train_labels = np.concatenate((y_vals[int(0*subarray_length):int(j*subarray_length)],y_vals[int((j+1)*subarray_length):]))
                                val_labels =  y_vals[int(j*subarray_length):int((j+1)*subarray_length)]
                            else:
                                train_array = x_vals[0:(k-1)*subarray_length]
                                val_array = x_vals[(k-1)*subarray_length:]
                                train_labels = y_vals[0:(k-1)*subarray_length]
                                val_labels = y_vals[(k-1)*subarray_length:]
                    
                            scores = Neural_Net(lr,4,layer_sizes,bs,ep,lb,dp, train_array,train_labels, 'relu', val_array,val_labels)
                            net_loss += scores[0]   #scores[0] contains the loss
                        #storing the model performance details in the dictionary
                        models[(lr,4,bs,ep,lb)] = net_loss 
                        val_losses.append(net_loss)
                        if (net_loss<least_loss):
                            best_lr = lr
                            best_bs =bs
                            best_num_layers=4
                            least_loss = net_loss
                            
                    if n == 5:
                        layer_sizes = [784,500,250,100,10]
                        for j in range(k):
                            if(j!= k-1):
                                train_array = np.concatenate((x_vals[int(0*subarray_len):int(j*subarray_len)],x_vals[int((j+1)*subarray_len):]))
                                val_array = x_vals[int(j*subarray_len):int((j+1)*subarray_len)]
                                train_labels = np.concatenate((y_vals[int(0*subarray_len):int(j*subarray_len)],y_vals[int((j+1)*subarray_len):]))
                                val_labels =  y_vals[int(j*subarray_len):int((j+1)*subarray_len)]
                            else:
                                train_array = x_vals[0:(k-1)*subarray_len]
                                val_array = x_vals[(k-1)*subarray_len:]
                                train_labels = y_vals[0:(k-1)*subarray_len]
                                val_labels = y_vals[(k-1)*subarray_len:]
                    
                            scores = Neural_Net(lr,5,layer_sizes,bs,ep,lb,dp, train_array,train_labels,'relu', val_array,val_labels)
                            net_loss += scores[0]   #scores[0] contains the loss
                            #storing the model performance details in the dictionary
                        models[(lr,5,bs,ep,lb)] = net_loss 
                        val_losses.append(net_loss)
                        if (net_loss<least_loss):
                            best_lr = lr
                            best_bs =bs
                            best_num_layers=5
                            least_loss = net_loss
        
                    print('reg %e, lr %e, n_layers %d, batch_size %d, epochs %d,  val loss: %f' % (lb,lr,5,bs,ep, net_loss))
        
    return (least_loss, best_model, models, best_lr, best_bs, best_num_layers)  

#least_loss, best_model, models, lr, bs, num_layers= cross_validate(3)


# In[ ]:




# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

num_classes = 10
#importing the keras elements
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Input,Flatten,Dense,Dropout
#from keras.layers.advanced_activations import LeakyReLu
from keras.optimizers import SGD
from keras.utils import to_categorical

data = pd.read_csv('2014EE10421.csv', header = None)
x = data.values[:, :784]
y = data.values[:, 784]
x = x/255
length = len(y)
x_train = x[:int(3*length/(4)), :]
x_val = x[int(3*length/4):, :]
l1= len(x_train[:, 0])
l2= len(x_val[:, 0])
x_train = x_train.reshape(l1, 28, 28, 1)
x_val = x_val.reshape(l2, 28, 28, 1)
y = to_categorical(y)
y = y.reshape(len(y[:, 0]), 10)
y_val = y[int(3*length/4):]
y_train = y[:(int(3*length/4))]
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# #model.add(Flatten())
# #model.add(Dense(10, activation=’softmax’))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Dropout(0.25))
# model.add(Conv2D(128,kernel_size = (3,3), activation = 'relu'))
# model.add(Conv2D(256,kernel_size = (3,3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128,activation = 'relu'))
model.add(Dense(250,activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])
model.summary()
model3fit = model.fit(x_train, y_train, batch_size=100, verbose=1, epochs=25)
score = model.evaluate(x_val, y_val)
print(score)

