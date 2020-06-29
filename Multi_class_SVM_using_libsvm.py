#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:14:53 2019

@author: Ritvik Sharma
"""



# In[ ]:


import pandas as pd
import numpy as np
from sklearn import svm
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.colors as col


class multi_class_svm:
    def __init__(self, k, c, degree, g, xtrain_data, ytrain_data, xtest_data, ytest_data):
        self.kernel = k
        self.c_val = c
        self.d0 = degree
        self.gam = g
        self.X_train = xtrain_data
        self.Y_train = ytrain_data
        self.X_test = xtest_data
        self.Y_test = ytest_data
        
        
    def svm_run(self):
        x_train = self.X_train
        y_train = self.Y_train
        x_test = self.X_test
        y_test = self.Y_test
        g = self.gam
        ker = self.kernel
        d = self.d0
        clf = svm.SVC(C = self.c_val, kernel = ker, gamma  = g, degree = d)
        clf.fit(x_train,y_train)
        predictions = clf.predict(x_test)
        correct_predictions = np.sum(predictions == y_test)
        accuracy = (correct_predictions/len(predictions)) 
        print(accuracy)
        return accuracy
    
    def test_new_prediction(x_train, y_train, x_test, y_test):
        g  = self.gam
        d = self.d0
        c_val = self.c_val
        ker = self.kernel
        clf = svm.SVC(C = self.c_val, kernel = ker, gamma  = g, degree = d)
        clf.fit(X_train,Y_train)
        predictions = clf.predict(x_test)
        return predictions
        
            
def cross_validate(k0):
    data = pd.read_csv("train_set.csv", header = None)
    a = data.values[:, 0]
    small_len = len(a)
    small_len = small_len/k0
    x_vals = data.values[:, :len(data.values[0, :])-1]
    y_vals = data.values[:, len(data.values[0, :])-1:len(data.values[0, :])]
    y_vals = y_vals.reshape(len(y_vals), )
    t =0
    ab = max(abs(np.amax(x_vals)), abs(np.amin(x_vals)))
    
    #### Following used to switch from multi class classifier to bianry classification
#     for i in range(len(y_val)):
#         if y_val[i]==0 or y_val[i] ==1:
#             t+=1
    #x_vals = np.zeros((t,len(data.values[0, :])-1))
    #y_vals = np.zeros(t)
    #t1 = 0
#     for i in range(len(y_val)):
#         if y_val[i] == 0 or y_val[i]==1:
#             x_vals[t1] = x_val[i]
#             y_vals[t1] = y_val[i]
#             t1+=1
#     small_len = t
#     small_len = small_len/k0 
    accuracy = 0
    x_vals=x_vals/ab ################## Normalization may or may not be used
    c_vals = [1e-3, 1e-1,0.5,1, 1.66, 1e1, 1e2]
    kernels = [ 'rbf', 'poly','linear']#,
    Z0 = np.zeros((len(c_vals), 7))
    Z1 = np.zeros((len(c_vals), 4))
    Z2 = np.zeros(len(c_vals))
    best_accuracy= 0
    s = -1
    for c in c_vals:
        s = s+1
        for k in kernels:
            if k == 'rbf':
                j=-1
                for g in [2.5, 1.66, 1, 0.75, 0.5, 0.25]:
                    j = j+1
                    accuracy = 0
                    for i in range(k0):
                        if i != k0-1:
                            x_train = np.concatenate((x_vals[int(0*small_len):int(i*small_len)],x_vals[int((i+1)*small_len):]))
                            y_train = np.concatenate((y_vals[int(0*small_len):int(i*small_len)],y_vals[int((i+1)*small_len):]))
                            x_test = x_vals[int((i)*small_len):int((i+1)*small_len)]
                            y_test = y_vals[int((i)*small_len):int((i+1)*small_len)]
                                
                        else:
                            x_train = x_vals[int(0*small_len):int((k0-1)*small_len)]
                            y_train = y_vals[int(0*small_len):int((k0-1)*small_len)]
                            x_test = x_vals[(int) ((k0-1)*small_len):]
                            y_test = y_vals[(int) ((k0-1)*small_len):]
                                
                        model = multi_class_svm(k, c, 0, g, x_train, y_train, x_test, y_test)
                        accuracy += model.svm_run()
                    accuracy = accuracy/4
                    Z0[s, j] = accuracy
                    if accuracy> best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                        
                        
            elif k == 'poly':
                g = 'auto'
                m=-1
                #print("hello", i)
                for d in [3, 4, 5, 7]:
                    m= m+1
                    accuracy = 0
                    for i in range(k0):
                        if i != k0-1:
                            x_train = np.concatenate((x_vals[int(0*small_len):int(i*small_len)],x_vals[int((i+1)*small_len):]))
                            y_train = np.concatenate((y_vals[int(0*small_len):int(i*small_len)],y_vals[int((i+1)*small_len):]))
                            x_test = x_vals[int((i)*small_len):int((i+1)*small_len)]
                            y_test = y_vals[int((i)*small_len):int((i+1)*small_len)]
                                
                        else:
                            x_train = x_vals[int(0*small_len):int((k0-1)*small_len)]
                            y_train = y_vals[int(0*small_len):int((k0-1)*small_len)]
                            x_test = x_vals[(int) ((k0-1)*small_len):]
                            y_test = y_vals[(int) ((k0-1)*small_len):]
                                
                        model = multi_class_svm(k, c, d, g, x_train, y_train, x_test, y_test)
                        accuracy += model.svm_run()
                    accuracy = accuracy/4
                    Z1[s, m] = accuracy
                    print("hello2", m, " ", accuracy)
                    if accuracy> best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                   
                        
            else:
                print("linear")
                g = 'auto'
                d = 0
                accuracy = 0               
                for i in range(k0):
                    if i != k0-1:
                        x_train = np.concatenate((x_vals[int(0*small_len):int(i*small_len)],x_vals[int((i+1)*small_len):]))
                        y_train = np.concatenate((y_vals[int(0*small_len):int(i*small_len)],y_vals[int((i+1)*small_len):]))
                        x_test = x_vals[int((i)*small_len):int((i+1)*small_len)]
                        y_test = y_vals[int((i)*small_len):int((i+1)*small_len)]
                                
                    else:
                        x_train = x_vals[int(0*small_len):int((k0-1)*small_len)]
                        y_train = y_vals[int(0*small_len):int((k0-1)*small_len)]
                        x_test = x_vals[(int) ((k0-1)*small_len):]
                        y_test = y_vals[(int) ((k0-1)*small_len):]
                                
                    model = multi_class_svm(k, c, d, g, x_train, y_train, x_test, y_test)
                    accuracy += model.svm_run()
                        #accuracy += svm_train_tester(x_train, y_train, x_test, y_test)
                accuracy = accuracy/4
                Z2[s] = accuracy
                if accuracy> best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                
    colors = ['red','green','blue','purple']
    labels = ["degree = 3","degree = 4" ,"degree = 5" ,"degree = 7"]
    #cdict = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}
    print(best_accuracy)
    print(best_model.c_val)
    print(best_model.gam)
    print(best_model.d0)
    plt.figure()
    print(Z0)
    
#     plt.cotour(np.log(np.asarray(c_vals)), np.asarray([10, 5, 2.5, 1.66, 1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001, 0.00001, 0.000000001]), Z0.T)
#     plt.show()
    #print(best_d)
    #print(best_predictions)
    return best_model, Z0, c_vals


#model, Z1, c_vals = cross_validate(4)
        


# In[ ]:


### Plotting

# cr, dr = np.meshgrid([-5, -3, -1, -0.3, 0, 0.22, 1, 2], (np.asarray([2.5, 1.66, 1, 0.75, 0.5, 0.25, 0.1])))
# fig, ax = plt.subplots()
# cs = ax.contourf(cr, dr, Z1.T)
# cbar = fig.colorbar(cs)
# cr = np.asarray([1e-1,0.5,1, 1.66, 1e1, 1e2, 1e3])
# plt.show()


# import matplotlib.pyplot as plt
# print(Z1.shape)
# plt.plot(np.asarray(np.asarray(np.log(c_vals))), Z2)
# plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn import svm
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.colors as col
data = pd.read_csv("train_set.csv", header = None)
a = data.values[:, 0]
x_vals = data.values[:, :len(data.values[0, :])-1]
y_vals = data.values[:, len(data.values[0, :])-1:len(data.values[0, :])]
y_vals = y_vals.reshape(len(y_vals), )
abt = max(abs(np.amax(x_vals)), abs(np.amin(x_vals)))
x_vals= x_vals/abt
abt = max(abs(np.amax(x_vals)), abs(np.amin(x_vals)))
clf = svm.SVC(C = 100, kernel = 'rbf', gamma  = 2.5)
print(x_vals[0:10, :])
print(y_vals[0:10])
clf.fit(x_vals, y_vals)
data_test = pd.read_csv("test_set.csv", header=None)
e = data_test.values[:, :]
x_test = data_test.values[:, :]
abt = max(abs(np.amax(x_test)), abs(np.amin(x_test)))
x_test= x_test/abt
predictions = clf.predict(x_test)
print(predictions.shape)
predictions = predictions.reshape(len(predictions), 1)
# #b = np.concatenate((x_test, predictions), axis = 1)
# print(np.arange(2000))
c = np.arange(2000)
c=c.reshape(len(c), 1)
# print(c.shape)
# print(predictions.shape)
b = np.concatenate((c, predictions), axis = 1)
print(b.shape)
print(b)
np.savetxt('foo4.csv', b, delimiter = ',',fmt ='%d,%d', header="id,class" )
data = pd.read_csv("foo4.csv")
print(data.values[:, :])
print(data.values[:, :].shape)


