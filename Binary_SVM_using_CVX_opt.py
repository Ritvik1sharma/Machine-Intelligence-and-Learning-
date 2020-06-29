#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:14:53 2019

@author: Ritvik Sharma
"""



# In[ ]:


from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
def hard_margin_linear(x_vals, y_vals):
    x_vals = np.concatenate([x_vals[y_vals == 0], x_vals[y_vals == 1]])
    y_vals = np.concatenate([y_vals[y_vals == 0], y_vals[y_vals == 1]])
    x_vals,y_vals = shuffle(x_vals,y_vals,random_state = 0)
    y_vals = 2*y_vals - 1    
    x_vals = x_vals.reshape(x_vals.shape[0],25)
    y_vals = y_vals.reshape(-1,1)*1
    tr_len = int(0.7*len(y_vals))
    x_train = x_vals[:tr_len, :]
    y_train = y_vals[0:tr_len, :]
    x_test = x_vals[tr_len:, :]
    y_test = y_vals[tr_len:, :]
    
    
    m,n = x_train.shape
#     x_dash = x_train*y_train
#     H = x_dash@x_dash.T
    X_dash = y_train* x_train
    H = np.dot(X_dash , X_dash.T) * 1.

    #X_dash = y_train*x_train
    #H = np.dot(X_dash , X_dash.T) * 1.
    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y_train.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters (change default to decrease tolerance) 
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])   
    #w parameter in vectorized form
    w = ((y_train*alphas).T @ x_train).reshape(-1,1)
    #Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()
    #Computing b
    b = y_train[S] - np.dot(x_train[S], w)
    #Display results
#     print('Alphas = ',alphas[alphas > 1e-4])
#     print('w = ', w.flatten())
#     print('b = ', b[0])
    scores = x_test.dot(w) + b[0]
    predictions = np.sign(scores)
    correct_predictions = np.sum(predictions == np.rint(y_test))
    val_accuracy = (correct_predictions/len(predictions))*100
    #print('validation_accuracy = ', val_accuracy)
    return val_accuracy
    
def soft_margin_linear(C, x_vals, y_vals):
    x_vals = np.concatenate([x_vals[y_vals == 0], x_vals[y_vals == 1]])
    y_vals = np.concatenate([y_vals[y_vals == 0], y_vals[y_vals == 1]])
    x_vals,y_vals = shuffle(x_vals,y_vals,random_state = 0)
    y_vals = 2*y_vals - 1    
    x_vals = x_vals.reshape(x_vals.shape[0],25)
    y_vals = y_vals.reshape(-1,1)*1
    tr_len = int(0.75*len(y_vals))
    x_train = x_vals[:tr_len, :]
    y_train = y_vals[0:tr_len, :]
    x_test = x_vals[tr_len:, :]
    y_test = y_vals[tr_len:, :]
    
    
    m,n = x_train.shape
#     x_dash = x_train*y_train
#     H = x_dash@x_dash.T
    X_dash = y_train* x_train
    H = np.dot(X_dash , X_dash.T) * 1
    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y_train.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters (change default to decrease tolerance) 
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    
    w = ((y_train*alphas).T @ x_train).reshape(-1,1)
    #Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()
    #Computing b
    b = y_train[S] - np.dot(x_train[S], w)
#     print('Alphas = ',alphas[alphas > 1e-4])
#     print('w = ', w.flatten())
#     print('b = ', b[0])
    scores = x_test.dot(w) + b[0]
    predictions = np.sign(scores)
    correct_predictions = np.sum(predictions == y_test)
    val_accuracy = (correct_predictions/len(predictions))*100
    #print('validation_accuracy = ', val_accuracy)
    return val_accuracy

def soft_margin_rbf(C, gamma, x_vals, y_vals):   
    x_vals = np.concatenate([x_vals[y_vals == 0], x_vals[y_vals == 1]])
    y_vals = np.concatenate([y_vals[y_vals == 0], y_vals[y_vals == 1]])
    x_vals,y_vals = shuffle(x_vals,y_vals,random_state = 0)
    y_vals = 2*y_vals - 1    
    x_vals = x_vals.reshape(x_vals.shape[0],25)
    y_vals = y_vals.reshape(-1,1)*1
    
    tr_len = int(0.7*len(y_vals))
    train_length=tr_len
    x_train = x_vals[:tr_len, :]
    y_train = y_vals[:tr_len, :]
    x_test = x_vals[tr_len:, :]
    y_test = y_vals[tr_len:, :]
    m,n = x_train.shape
    norm_column_squared = np.linalg.norm(x_train, axis = 1).reshape(m,1)**2
    norm_row_squared = np.linalg.norm(x_train.T, axis = 0).reshape(1,m)**2    
    norm_column_extended = norm_column_squared*np.ones((m,m))
    norm_row_extended = norm_column_extended.T
    cross_terms = x_train@x_train.T
    
    #rbf matrix has terms of the form aij = exp(-gamma*||xi - xj||^2)
    X_dash = np.exp(-gamma*(norm_column_extended + norm_row_extended - 2*cross_terms))
    #((y_train.reshape(tr_len,1)*X_dash).T)
    H = (y_train.reshape(tr_len,1))*((y_train.reshape(tr_len,1)*X_dash).T)        
    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y_train.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    
    #Setting solver parameters (change default to decrease tolerance) 
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    
    w = ((y_train*alphas).T @ x_train).reshape(-1,1)
    #Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()
    #Computing b
    b = y_train[S] - np.dot(x_train[S], w)
    #Display results
#     print('Alphas = ',alphas[alphas > 1e-4])
#     print('w = ', w.flatten())
#     print('b = ', b[0])
    scores = x_test.dot(w) + b[0]
    predictions = np.sign(scores)

    correct_predictions = np.sum(predictions == y_test)
    val_accuracy = (correct_predictions/len(predictions))*100
    return val_accuracy
    
    
def hard_margin_rbf(gamma, x_vals, y_vals):   
    x_vals = np.concatenate([x_vals[y_vals == 0], x_vals[y_vals == 1]])
    y_vals = np.concatenate([y_vals[y_vals == 0], y_vals[y_vals == 1]])
    x_vals,y_vals = shuffle(x_vals,y_vals,random_state = 0)
    y_vals = 2*y_vals - 1    
    x_vals = x_vals.reshape(x_vals.shape[0],25)
    y_vals = y_vals.reshape(-1,1)*1
    
    tr_len = int(0.7*len(y_vals))
    x_train = x_vals[:tr_len, :]
    y_train = y_vals[:tr_len, :]
    #print(y_vals.shape)
    #print(y_train.shape)
    x_test = x_vals[tr_len:, :]
    y_test = y_vals[tr_len:, :]
    m,n = x_train.shape
    norm_column_squared = np.linalg.norm(x_train, axis = 1).reshape(m,1)**2
    norm_row_squared = np.linalg.norm(x_train.T, axis = 0).reshape(1,m)**2    
    norm_column_extended = norm_column_squared*np.ones((m,m))
    norm_row_extended = norm_column_extended.T
    cross_terms = x_train@x_train.T
    X_dash = np.exp(-gamma*(norm_column_extended + norm_row_extended - 2*cross_terms))
    #((y_train.reshape(tr_len,1)*X_dash).T)
    H = (y_train.reshape(tr_len,1))*((y_train.reshape(tr_len,1)*X_dash).T) 
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m,1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y_train.reshape(1,-1))
    b = cvxopt_matrix(np.zeros(1))    
    #Setting solver parameters (change default to decrease tolerance) 
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    
    w = ((y_train*alphas).T @ x_train).reshape(-1,1)
    #Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()
    #Computing b
    b = y_train[S] - np.dot(x_train[S], w)
    #Display results
#     print('Alphas = ',alphas[alphas > 1e-4])
#     print('w = ', w.flatten())
#     print('b = ', b[0])
    scores = x_test.dot(w) + b[0]
    predictions = np.sign(scores)
    correct_predictions = np.sum(predictions == y_test)
    val_accuracy = (correct_predictions/len(predictions))*100
    return val_accuracy

    
def validation_cvx_opt():    
    gammas = [1e-5,1e-19,1e-18,1e-17,1e-16,1e-15]
    c_vals = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6]
    data = pd.read_csv('2017EE10485.csv', header = None)
    x_vals = data.values[:, 0:25]
    y_vals = data.values[:, 25]
    print(y_vals.shape)
    for c in c_vals:
        for g in gammas:
            val_acc = hard_margin_linear(x_vals, y_vals)
            print('C: %e, validation_accuracy: %f' %(c, val_acc))
    return


        
   


# In[ ]:


#validation_cvx_opt()

