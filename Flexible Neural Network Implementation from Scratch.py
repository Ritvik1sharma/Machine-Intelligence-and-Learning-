#!/usr/bin/env python
# coding: utf-8

# In[28]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:



import numpy as np
import pandas as pd

class NeuralNetwork1:        
    def __init__(self,x_train, y_train, x_val, y_val, lr,epochs,num_layers,layer_sizes,lb, batch):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_val
        self.y_test = y_val
        self.lr= lr
        self.epoch = epochs
        self.lamb = lb
        self.inputSize = (len(x_train[0]))
        self.outputSize = 10
        #self.hiddenSize = hiddensize
        self.ls = layer_sizes
        
        self.num_layers =num_layers 
        self.batch = batch
        #layer_sizes = hiddensize
        ws = []
        
        self.bs = []        
        for i in range(num_layers - 1):
            ws.append(0.1*np.random.randn(layer_sizes[i+1],layer_sizes[i]))
            self.bs.append(0.1*np.random.randn(layer_sizes[i+1],1))
        self.num_layers = num_layers
        self.W = ws
        
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_prime(self,x):
        
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    def softmax(self,x):
        exp_vec = np.exp(x)
        softmax = exp_vec/np.sum(exp_vec,axis = 0)
        return softmax
        
    def relu(self,x):
        #return np.maximum(x,0)
        return (np.maximum(x,0)+0.01*np.minimum(x, 0))
    
    def relu_prime(self,x):
        #return 0.5*np.sign(x) + 0.5
        return 0.505*np.sign(x) +0.495  #leaky relu prime
    
        
    def forward(self, x):
        z1 = []
        w = self.W
        t = x
        #print(x.shape)
        z2 = [x]
        for i in range((self.num_layers)-1):
            if i ==0:
                t = self.W[i].dot(x) + self.bs[i]
                z1.append(t)               
                t = self.relu(t)
                z2.append(t)
            elif i!= self.num_layers - 2:
                t = self.W[i].dot(t) + self.bs[i]
                z1.append(t)               
                t = self.relu(t) 
                z2.append(t)
            else:
                t = self.W[i].dot(t) + self.bs[i]
                z1.append(t)
                t = self.softmax(t) # activation function
                z2.append(t)     
            #print("helloworld", z2[i].shape)
        return z1, z2 
    
    def backPropagation(self,x,y,lb = 0):        
        N = x.shape[1]
        (z_1,z_2) = self.forward(x)
        dw = []
        db = []
        delta = [None]*len(self.W)
        softmax_p = z_2[-1]
        for i in range(N):
            softmax_p[y[i], i] += -1
        delta[-1] = softmax_p/N
        delta[-1] = delta[-1].reshape(delta[-1].shape[0],N)
        for i in reversed(range(len(delta) - 1)):
            delta[i] = ((self.W[i+1].T.dot(delta[i+1]))*(self.relu_prime(z_1[i])))    
        batch_size = x.shape[1]
        dw = [d.dot(z_2[i].T)/float(batch_size) +lb*self.W[i]/float(batch_size) for i,d in enumerate(delta)]
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in delta]
        return dw,db
       
    def train(self):
        batch_size = self.batch
        epochs = self.epoch
        lr= self.lr
        lb = self.lamb
        x = self.x_train
        y = self.y_train
        for ep in range(epochs):
            print("epochs : ", ep)
            i = 0
            while(i<len(y)):
                x_batch = x[:,i:i+batch_size]
                y_batch = y[i:i+batch_size]
                N = self.batch
                i = i+batch_size
                (z_1, z_2) = self.forward(x_batch)
                (dw,db) = self.backPropagation(x_batch,y_batch, self.lamb)
                
                self.W = [w - lr*dweight for w,dweight in zip(self.W,dw)]
                self.bs = [w - lr*dbias for w,dbias in zip(self.bs, db)]
        return self.W             

    def test(self):
        x_batch = self.x_test
        y_batch = self.y_test
        N = len(y_batch)
        (z_1, z_2) = self.forward(x_batch)
        scores = z_2[-1]
        predictions = np.argmax(scores, axis = 0)
        acc = np.sum(predictions == y_batch)/len(y_batch)
        scores2= np.zeros(N)
        for i in range(N):
            scores2[i] = scores[y_batch[i], i]
        loss = np.sum(-np.log(scores2 + 1e-30))
        loss/= N
        x_t = self.x_train
        y_t = self.y_train
        N = len(y_t)
        (z1, z2) = self.forward(x_batch)
        final_scores = z2[-1]
        predictions = np.argmax(final_scores, axis = 0)
        a1 = np.sum(predictions == y_batch)/len(y_batch)
        print("Train acc: ", a1)
        return acc, loss

def cross_validation(k):
    data = pd.read_csv('2014EE10421.csv', header = None)
    x_vals = data.values[0:,0:784]/255
    y_vals = data.values[0:,784]
    subarray_length = len(x_vals)//k

    best_W= None
    alphas = [0.5, 1.0]#0.001, 0.05, 
    epochs = [25]
    batch_sizes = [8]
    num_layers = [5]
    lambdas = [1e-14, 1e-7, 0.00001, 0.01]
    bestloss = float('inf') 
    best_model = None
    small_len = (len(x_vals))//k
    
    x_vals = x_vals.reshape((len(x_vals), 784))
    least_loss = float('inf')
    best_params = None
    
    models = {}
    val_losses = []
    for lr in alphas:
        for ep in epochs:
            for bs in batch_sizes:
                for l in lambdas:
                    for num in num_layers:
                        net_loss = 0
                        acc_avg = 0
                        lb = l
                        if num == 3:
                            layers = [784, 25, 10]
                            for i in range(k):
                                if i != k-1:            
                                    x_input = np.concatenate((x_vals[int(0*small_len):int(i*small_len)],x_vals[int((i+1)*small_len):]))
                                    y_input = np.concatenate((y_vals[int(0*small_len):int(i*small_len)],y_vals[int((i+1)*small_len):]))
                                    x_test = x_vals[int((i)*small_len):int((i+1)*small_len)]
                                    y_test = y_vals[int((i)*small_len):int((i+1)*small_len)]
                                
                                else:
                                    x_input = x_vals[int(0*small_len):int((k-1)*small_len)]
                                    y_input = y_vals[int(0*small_len):int((k-1)*small_len)]
                                    x_test = x_vals[(int) ((k-1)*small_len):]
                                    y_test = y_vals[(int) ((k-1)*small_len):]

                                model = NeuralNetwork1(x_input.T, y_input, x_test.T, y_test, lr, ep, 3, layers, l, bs)
                                W= model.train()
                                acc, loss = model.validate()
                                net_loss += loss
                                acc_avg += acc
                                print(acc)
                            net_loss = net_loss/k
                            acc_avg = acc_avg/k
                            models[(lr,num,bs,lb)] = (net_loss,acc_avg)
                            val_losses.append(net_loss)                     
                            print('reg %e, lr %e, n_layers %d, batch_size %d, epochs %d,  val loss: %f, val_accuracy: %f' % (lb,lr,num,bs,ep,net_loss, acc_avg))
                            if (net_loss<least_loss):
                                least_loss = net_loss
                                best_acc = acc_avg
                                best_params = (lr,num,bs,lb)
                                best_W = W

                        if num == 4:
                            layers = [784, 400, 100, 10]
                            for i in range(k):
                                if i != k-1:
                                    x_input = np.concatenate(( x_vals[int(0*small_len):int(i*small_len)] , x_vals[int((i+1)*small_len):] ))
                                    y_input = np.concatenate((y_vals[int(0*small_len):int(i*small_len)],y_vals[int((i+1)*small_len):]))
                                    x_test = x_vals[int((i)*small_len):int((i+1)*small_len)]
                                    y_test = y_vals[int((i)*small_len):int((i+1)*small_len)]
                                
                                else:
                                    x_input = x_vals[int(0*small_len):int((k-1)*small_len)]
                                    y_input = y_vals[int(0*small_len):int((k-1)*small_len)]
                                    x_test = x_vals[(int) ((k-1)*small_len):]
                                    y_test = y_vals[(int) ((k-1)*small_len):]
                                model = NeuralNetwork1(x_input.T, y_input, x_test.T, y_test, lr, ep, 4, layers, l, bs)
                                W = model.train()
                                acc, loss = model.validate() #scores[0] contains the loss
                                net_loss += loss
                                acc_avg += acc
                            net_loss = net_loss/k
                            acc_avg = acc_avg/k
                            models[(lr,num,bs,lb)] = (net_loss,acc_avg)
                            val_losses.append(net_loss)                     
                            print('reg %e, lr %e, n_layers %d, batch_size %d, epochs %d,  val loss: %f, val_accuracy: %f' % (lb,lr,num,bs,ep,net_loss, acc_avg))
                            if (net_loss<least_loss):
                                least_loss = net_loss
                                best_acc = acc_avg
                                best_params = (lr,num,bs,lb)
                                best_W = W
                                
                        if num == 5:
                            layer_sizes = [784,500,300,100,10]
                            net_loss = 0
                            acc_avg = 0
                            for i in range(k):
                                if(i!= k-1):
                                    x_input = np.concatenate((x_vals[int(0*subarray_length):int(i*subarray_length)],x_vals[int((i+1)*subarray_length):]))
                                    x_test = x_vals[int(i*subarray_length):int((i+1)*subarray_length)]
                                    y_input = np.concatenate((y_vals[int(0*subarray_length):int(i*subarray_length)],y_vals[int((i+1)*subarray_length):]))
                                    y_test =  y_vals[int(i*subarray_length):int((i+1)*subarray_length)]
                    
                                else:
                                    x_input = x_vals[0:(k-1)*subarray_length]
                                    x_test = x_vals[(k-1)*subarray_length:]
                                    y_input = y_vals[0:(k-1)*subarray_length]
                                    y_test = y_vals[(k-1)*subarray_length:]
                    
                                model = NeuralNetwork1(x_input.T, y_input, x_test.T, y_test, lr, ep, 5,[784,500,250,100,10], l, bs)
                                W = model.train()
                                acc, loss = model.validate()
                                net_loss += loss
                                acc_avg += acc
                                print(acc)
                            net_loss = net_loss/k
                            acc_avg = acc_avg/k
                            models[(lr,num,bs,l)] = (net_loss,acc_avg)
                            val_losses.append(net_loss) 
                            if (net_loss<least_loss):
                                least_loss = net_loss
                                best_acc = acc_avg
                                best_params = (lr,num,bs,lb)
                                best_W = W
        
                            print('reg %e, lr %e, n_layers %d, batch_size %d, epochs %d,  val loss: %f, val_accuracy: %f' % (lb,lr,num,bs,ep,net_loss, acc_avg))   
    return (best_acc, least_loss,best_params, models)        

(acc, a,b,c) = cross_validation(4)


# In[ ]:




