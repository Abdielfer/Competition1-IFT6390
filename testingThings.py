#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:13:01 2021

@author: stremblay
"""
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import math
from math import exp
from sklearn.preprocessing import OneHotEncoder # One can do it by hand too..
onehot_encoder = OneHotEncoder(sparse=False)
from scipy.special import expit
 
###Data
train_Set = pd.read_csv("train.csv")

train_col_names = train_Set.columns
trainig_y = train_Set.loc[:, 'LABELS']
unique_label = np.unique(trainig_y)
# build trainig_x set removing 'index' and 'labels' column from origin
trainig_x = train_Set.drop([train_col_names[0], train_col_names[-1]], axis=1)
# add bias column to training set
bias = np.ones((trainig_x.shape[0]))
trainig_x.insert(0, 'bias', bias)

# Test_Set.info()
test_Set = pd.read_csv("test.csv")
# add bias column to test set
test_Set = test_Set.drop([train_col_names[0]], axis=1)
bias = np.ones((test_Set.shape[0]))
test_Set.insert(0, 'bias', bias)

x = trainig_x.loc[0:999,:]
Y = trainig_y[0:1000]
Y = Y.iloc[:]
w = np.random.rand(unique_label.shape[0],x.shape[1])/1000


      
# Logistic regression

class logisticRegression():
    def __init__(self,x,y,alpha,epochs):
        self.w = np.random.rand(np.unique(y).shape[0], x.shape[1])/100
        self.x = x
        self.Y = y
        self.y_labels = np.unique(y)
        self.alpha = alpha
        self.epochs = epochs
         
    def sigmoid(self,w,x): ## Devuelve probabilidad de apartenancia a una clase 
        z = np.array(np.dot(w, x.T))
        z = expit(z)
        h = 1 / (1 + np.exp(-z))
        return np.array(h)  ## h (class,samples)
    
    def hypotesis(self,theta,x):
        activationFunction = self.sigmoid(theta,x)
        return activationFunction
    
    def prediction(self,hypotesis): 
        '''
        return de predicted classes from the hypotesis function result (sigmoid(W,X))
        '''
        y_hat = []    
        for i in hypotesis:
            y_hat.append(np.argmax(i))
        y_hat = np.array(y_hat)
        return y_hat
    
    def prediction_probab(self,z): 
        '''
        return devector(features,1) of probailiti of predicted classes from the hypotesis function
            result (sigmoid(W,X))
        @z: it's a matrix(classes,features) of probabilities z = sigmoind(W,X)  
        '''
        y_hat_prob = []
        for i in range(z.shape[0]):
            y_hat_prob.append(np.argmax(z[i]))
        return np.array(y_hat_prob)
     
    def cost(self,Y,y_hat_prob):
        '''  
        cross entropy loss function
        cost = -1/m * sum_(1-m){y_hat*log(y) + (1-y_hat)log(1-y)}  
        @z: it's a vector of probabilities z = sigmoind() (non zero values plc)  
        @Y: real classes
        + 0.000001 to avoid 0 divition & log(0) 
        '''
        '''
        plan B
        m = len(Y)
        error = 0
        y = np.array(Y) #ensure Y is in array form
        cummulative = 0
        for i in range(m-1):
            #cummulative += y_hat_prob[i] - y[i]
            cummulative += (y[i]*np.log(y_hat_prob[i])) + (1-y[i])*np.log(1-y_hat_prob[i])
        error = -np.sum(cummulative, axis = 0)**2
        return error/m
        '''
        m = len(Y)
        cummulative = 0
        for i in range(m):
            cummulative += y_hat_prob[i] - Y[i]
        sqr_error = np.sum(cummulative, axis = 0)**2 
        return sqr_error/m
    
    def crossEntropLoss(self,Y_onehot,z):
        '''  
        cross entropy loss function's derivative respect w_feature
        costDerivativeRespectFeuature = -1/m * sum_(1-m){(z-y)*X_feature.T}
        @z: k class column (1...k) of the vector of probabilities z = sigmoind() 
        @Y_onehot: k class column(1...k) of Actual values from the y_onehot_encoded matrix
        @x_ features: j column (1.. n_features) from DataSet
        '''
        m = len(Y_onehot)
        gradient = (1/m)*(z-Y_onehot)
        return gradient
       
    def crossEntropLoss_derivative(self,Y_onehot,z,x_feature):
        '''  
        cross entropy loss function's derivative respect w_feature
        costDerivativeRespectFeuature = -1/m * sum_(1-m){(z-y)*X_feature.T}
        @z: k class column (1...k) of the vector of probabilities z = sigmoind() 
        @Y_onehot: k class column(1...k) of Actual values from the y_onehot_encoded matrix
        @x_ features: j column (1.. n_features) from DataSet
        '''
        m = len(Y_onehot)
        gradient = (1/m)*np.dot((z-Y_onehot),x_feature)
        return gradient
    
    def error(self,y_hat, y):
        """Return the error rate for a batch X.
        X is a matrix of size n_examples x n_features.
        y is an array of size n_examples.
        Returns a scalar.
        """  
        n = len(y_hat)
        cont = 0
        for i in range(n):
            if y_hat[i] != y[i]:
                  cont+=1 
        return cont/n
    
    def gradientDescent(self,X,Y):
        m = len(X) # number of samles
        theta_gd = self.w.copy().astype('float32')
        loss = []
        errors = []
        cost = 0
        Y_onehot = onehot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
        for i in range(self.epochs):
            z = self.hypotesis(theta_gd,X)
            y_hat_prob = self.prediction_probab(z.T)
            globalCost = self.cost(self.Y,y_hat_prob)
            loss.append(globalCost)
            y_hat = self.prediction(z.T)
            global_error = self.error(y_hat,Y)
            errors.append(global_error)
            print(globalCost, global_error)
            ## updating theta
            for k in range(len(self.y_labels)): ## Classes w_shape=(Classes, features)
                for j in range(X.shape[1]): ## Features  X_ahpe = (samples, features)
                    if j == 1:
                        x_feature = X.iloc[:,j]
                        cost = self.crossEntropLoss(Y_onehot[:,k],z[k,:].T)
                        theta_gd[k,j] -= self.alpha*cost
                    else:    
                        x_feature = X.iloc[:,j]
                        cost_deriv = self.crossEntropLoss_derivative(Y_onehot[:,k],z[k,:].T,x_feature)
                        theta_gd[k,j] -= self.alpha*cost_deriv
         
            # graphyc lost and error  ....
         
        return theta_gd, np.array(loss), np.array(errors)
 

lr=logisticRegression(x,Y,0.03,50)
theta_gd, loss, errors = lr.gradientDescent(trainig_x, trainig_y)





















