#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:58:00 2021

@author: A.Fernandez
"""
# Import
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import math, decimal
from math import exp
import sklearn as sk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder # One can do it by hand too..See Notes at the end of the code
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.special import expit  #  to prevent overflow in exp(-z)
import joblib

onehot_encoder = OneHotEncoder(sparse=False)


### Data manipulation and helper methods  ###
def meanStd(dataset):
    '''
    dataset_minmax(dataset)
    return a list like {min:#,max:#}
    # Find the min and std values for each column
    '''
    col = dataset.shape[1]
    meanVal, stdVal = 0,0
    stats = list()
    for i in range(col):
        val = dataset.iloc[:, i]
        meanVal = np.mean(val)
        stdVal = np.std(val)
        stats.append([meanVal,stdVal])
    return stats

def standardize_data(dataset, mean_std):
    '''
    standardize_data(dataset, mean_std)
    @mean_std: @arguent: list of min/max valuer per column {min:#,max:#}
    # Rescale dataset columns to the range 0-1
    '''
    col = dataset.shape[1]
    row = dataset.shape[0]
    for i in range(1,col):
        for n in range(row):
            dataset.iloc[n,i] -= mean_std[i][0]
            dataset.iloc[n,i] /= mean_std[i][1]
    return dataset

def newDataDomain_as_Array(listColName, dataSet):
    '''
    newDataDomain_as_Array(listCollName, dataSet)
    # return: subset of DataFrame (as numpy.ndarray()) from a list of columns name
    # @listCollName: @argument: is a list of collum's name from dataSet,
      # that you wnat to build the new data domain
    '''
    row = dataSet.shape[0]
    col = len(listColName)
    dataDomain = np.ndarray((row, col))
    i = 0
    for name in listColName:
        dataDomain[:, i] = dataSet.loc[:, name]
        i += 1
    return dataDomain

def newDataDomain_as_DFrame(listColName, dataSet):
    '''
    newDataDomain_as_DFrame(listCollName, dataSet)
    # return: A DataFrame with a subset of the DataFrame in "dataSet" with the columns in "listCollName"
    # @listCollName: @argument: is a list of columns name from the original dataSet,
    # @dataSet: @argument: The original dataSet
    '''
    df = pd.DataFrame(data=dataSet, columns=listColName)
    return df

def get_month_from_time(dataset, timeLabel):
    '''
    get_month_from_time(yyyymmdd)#
    return: month as Unique value with the corresponding ordinal number(i.e. Jan = 01,..., Dec=12) (int64)
    # @yyyymmdd: @argumen : DataFrame column containig date in format "yyyymmdd" (int64)
    '''
    for i in range(dataset.shape[0]):
        time = dataset.loc[i,timeLabel]
        m = abs((time % 10000)//100)
        dataset.loc[i,timeLabel] = m
    return dataset

def plot_gradientDescent(iters, loss, error, figureName):
    '''
    # plot_2D_fromDataFrame(df_x, df_y, title)
    # loss,error: @argument: vector as dataFrame
    '''
    ax = figureName.add_subplot(111)    # The big subplot
      # Cleaning the background
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax1 = figureName.add_subplot(211)
    ax2 = figureName.add_subplot(212)
    ax.set_xlabel('Epochs')
    ax1.set_title('Loss in Epochs')
    ax2.set_title('Errors in Epochs')
    ax1.plot(iters, loss)
    ax2.plot(iters, error)

    figureName.show

    return figureName, ax1, ax2

def updatePlot_gradDescent(itesr, loss, errors,f, subPlot1, subPlot2):
    '''Allows a real time update from the gradient descent evolution'''
    plt.ion
    subPlot1.clear()
    subPlot2.clear()
    subPlot1.plot(itesr, loss)
    subPlot2.plot(itesr, errors)
    plt.savefig('Loss_error_epochs.png', dpi=300)

# month = get_month_from_time(newDataDomain_as_DFrame(['time'], train_Set))

# Logistic regression (with regularization optional)
class logisticRegression():
    def __init__(self,x,y,lamb,alpha,epochs):
        self.w = np.zeros((np.unique(y).shape[0], x.shape[1]))  #random.rand  /100
        self.x = x
        self.y = y
        self.y_labels = np.unique(y)
        self.alpha = alpha
        self.lamb = lamb
        self.epochs = epochs

    def sigmoid(self,w,x):
        '''Sigmoid function computed with the theta matrix and the feature matrix
        @Return: h: a np.array shaped(k_class,n_samples) containig the probability of each sample to belong to class 1..d
        @ w: @argument: matrix of(k_clases, d_features)
        @ x: @argument: matrix of(n_samples,d_features)
        '''
        z = np.array(np.dot(w,x.T))
        z = expit(z)
        h = 1 / (1 + np.exp(-z))
        return np.array(h)  ## h (n_samples,k_class)

    def hypotesis(self,theta,x):
        '''Apply the sigmoid to the data, its a transition function to help code readability'''
        activationFunction = self.sigmoid(theta,x)
        return activationFunction

    def prediction(self,hypotesis):
        '''
        return de vector containig the predicted classes for each sample i.
        @y_hat[i]: The index of max probability found in hipotesis[i] and assigns it as the "Class" to the feature i
        NOTE: np.argmax(i): return the index of the max value, in our case it correspond with the class " 0,1,2"
        '''
        y_hat = []
        for i in hypotesis:
            y_hat.append(np.argmax(i))
        y_hat = np.array(y_hat)
        return y_hat

    def prediction_probab(self,hypotesis):
        '''
        return a vector(n_samples,) of probabiliti of predicted classes from the sigmoid function result hypotesis)
        @hypotesis: it's a matrix shaped(n_samples,k_class,) of probabilities computed throgh the activation function hypotesis = sigmoind(W,X)
        '''
        y_hat_prob = []
        for i in range(hypotesis.shape[0]):
            y_hat_prob.append(hypotesis[i][np.argmax(hypotesis[i])])
        return np.array(y_hat_prob)

    def cost(self,Y,y_hat_prob):  ###  VERIFICA ESTO RAPIDOOOOO... SI NO FUNCIONA DEJALO..
        '''
        return: Average error between real labels and predicted ones.
        @y_hat_prob: it's a vector of probabilities (non zero values plc...;))
        @Y: real classes
        '''
        #plan- B
        m = len(Y)
        cummulative = 0
        for i in range(m):
            cummulative += y_hat_prob[i] - Y[i]
        sqr_error = np.sum(cummulative, axis = 0)
        return sqr_error/m

        # m = len(Y)
        # error = 0
        # y = np.array(Y) #ensure Y is in array form
        # cummulative = 0
        # for i in range(m-1):
        #     #cummulative += y_hat_prob[i] - y[i]
        #     cummulative += (y[i]*np.log(y_hat_prob[i])) + (1-y[i])*np.log(1-y_hat_prob[i])
        # error = -np.sum(cummulative, axis = 0)

        #return error/m




    def crossEntropLoss_deriv4Bias(self,Y_onehot,z,x_feature):
        '''
        cross entropy loss function derivative respect bias column (= 1)
        costDerivativeRespectFeature[0] =1/m * sum_(1..m){(z-y)}
        @z: = hypotesis result (z for reduction in reading ...I'm lazy now..;)
        @Y_onehot: y_onehot_encoded matrix (n_samples, k_classes)
        @x_ features: j column (1.. n_features) from DataSet
        '''
        m = len(Y_onehot)  # m Samples
        gradient = (1/m)*np.sum(z-Y_onehot)
        return gradient


    def crossEntropLoss_derivative(self,Y_onehot,z,x_feature):
        '''
        cross entropy loss function's derivative respect w_feature
        costDerivativeRespectFeuature = -1/m * sum_(1-m){(z-y)*X_feature.T}
        @z: vector of k class column (1...k) from the matrix of probabilities z(hypotesis)
        @Y_onehot : vector k class column(1...k) of Actual values from the y_onehot_encoded matrix
        @x_feature : vector of feature d (n_samples,) from DataSet
        '''
        m = len(Y_onehot)
        gradient = (1/m)*np.dot((z-Y_onehot),x_feature)
        return gradient

    def regularisation(self,Y_onehot,z,x_feature,lamb, thetas_i):
        '''
        return: the regularized term for theta optimization
        1- compute crossEntropLoss_derivative(Y_onehot,z,x_feature) .. see description
        2- add the regularization term "(lamb/m)*thetas_i"
        @lamb : scalar: is the regularization factor
        @thetas_i : the theta value to be optimized
        '''
        gradient_reularized = self.crossEntropLoss_derivative(Y_onehot,z,x_feature)
        m = len(Y_onehot)
        gradient_reularized += (lamb/m)*thetas_i
        return gradient_reularized

    def error(self,y_hat, y):
        """Return the error rate for a batch X.
        y_hat & y: are arrays of size n_examples.
        Returns a scalar.
        """
        n = len(y_hat)
        cont = 0
        for i in range(n):
            A = y_hat[i]
            B = y[i]
            if A != B:
                  cont+=1
        return cont/n

    def gradientDescent(self):
        step = 0
        X = self.x
        Y = self.y
        m = len(X) # number of samles
        theta_gd = self.w.copy().astype('float32')
        Y_onehot = onehot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

        loss = []
        errors = []
        iters = []

        f = plt.figure()
        f,subPlot1, subPlot2 = plot_gradientDescent(iters, loss, errors,f)
        f.show()

        z = self.hypotesis(theta_gd,X)
        for i in range(self.epochs):
            ## updating theta
            globalCost =0
            for k in range(len(self.y_labels)): ## Classes w_shape=(k_Classes, d_features)
                for j in range(X.shape[1]): ## Features  X_shape = (n_samples,d_features)
                    x_feature = X.iloc[:,j]
                    theta_i = theta_gd[k,j]
                    if j==0:
                        # we regularize diferently theta_0
                        cost = self.crossEntropLoss_deriv4Bias(Y_onehot[:,k],z[k,:].T,x_feature)
                        theta_gd[k,j] -= self.alpha*cost
                    else:
                        #cost = self.crossEntropLoss_derivative(Y_onehot[:,k],z[k,:].T,x_feature)  # non regularized
                        cost = self.regularisation(Y_onehot[:,k],z[k,:].T,x_feature, self.lamb, theta_i)
                        theta_gd[k,j] -= self.alpha*cost

            z = self.hypotesis(theta_gd,X)
            y_hat_prob = self.prediction_probab(z.T)
            y_hat = self.prediction(z.T)
            globalCost = self.cost(Y,y_hat)
            loss.append(globalCost)
            global_error = self.error(y_hat,Y)
            errors.append(global_error)
            iters.append(i)
            print(globalCost, global_error, '__', i )

            ## Showing whats happend
            if i % 20 == 0:
                updatePlot_gradDescent(iters,loss, errors,f,subPlot1, subPlot2)
                f.show()

        return theta_gd, np.array(loss), np.array(errors)

class making_prediction():
    def __init__(self,x_test,thetas):
        self.w = thetas
        self.x = x_test
        #self.prediction = self.prediction(self.hypotesis(self.w,self.x).T)

    def sigmoid(self,w,x): ## The proability of been in one class
        z = np.array(np.dot(w, x.T))
        z = expit(z)
        h = 1 / (1 + np.exp(-z))
        return np.array(h)  ## h (n_samples,k_class)

    def hypotesis(self,theta,x):
        h = self.sigmoid(theta,x)
        return h

    def formatPrediction(self,hypotesis, y_hat):
        '''
        FOR the implemented from scratch LOGISTIC REGRESSION
        return the predicted classes with the desired format to kaggle submition
        @hypotesis : matrix of probablilities
        '''
        y_hat = pd.DataFrame({'S.No' : [],'LABELS' : []}, dtype=np.int8)
        for i in range(hypotesis.shape[0]):
            y_hat.loc[i] = [i,np.argmax(hypotesis[i])]
        return pd.DataFrame(data = y_hat)

    def formating_prediction(predictions):
        '''
        return de predicted classes from the hypotesis function result (sigmoid(W,X))
        @hypotesis : matrix of probablilities
        '''
        y_hat = pd.DataFrame({'S.No' : [],'LABELS' : []}, dtype=np.int8)
        for i in range(len(predictions)):
            y_hat.loc[i] = [i,predictions[i]]

        return pd.DataFrame(data = y_hat)

class otherModels():
    def __init__(self,x_set,y_set,test_set ):
         self.train_X = x_set
         self.train_Y = y_set
         self.testSet = test_set

    def svm(self, trainSet_x, train_y, testSet_x):
        '''
        Performe SVM from Skl library
        return classifier_SVM, y_hat_df
        return: model "classifier_SVM" and prediction "y_hat_df"in dataFrame format
        '''
        # ####  SVM
        classifier_SVM = SVC()
        classifier_SVM.fit(trainSet_x, train_y)
        ## MAKE AND Safe prediction
        y_hat = classifier_SVM.predict(testSet_x)
        y_hat_df = prediction(y_hat)
        ### uncomment next line to save you prediction
        # y_hat_df.to_csv('prediction_SVM_corretedSet.csv', index = False)

        print(classifier_SVM.score(trainSet_x, train_y),"____ SVM score")
         ## ........WE can do confusion matrix as well
        # confusion_matrix = confusion_matrix(train_y, y_hat_training)
        # print(confusion_matrix)
        print(classification_report(train_y, y_hat_training))
        return classifier_SVM, y_hat_df

    def logReg(self, trainSet_x, train_y, dalidatoin_x):
        '''
        Performe Logistic regression, in the simples way, from Skl library
        return classifier_LR
        return: model "classifier_LR"
        '''
        ###  LR
        classifier_LR = LogisticRegression(max_iter=1200)
        classifier_LR.fit(trainSet_x, train_y)
        print(classifier_LR.score(trainSet_x, train_y),"____ LR score")
        return classifier_LR

    def randomForest(train_set, labels, test_Set, estimators):
        '''
        Performe random forest, in the simples way, from Skl library
        return: model "classifier_LR" and a prediction over a test_set
        '''
        rf_classifier = RandomForestClassifier(n_estimators = estimators, criterion = 'entropy', random_state = 42)
        rf_classifier.fit(train_set, labels)

        # Predicting the Test set results
        y_pred = rf_classifier.predict(full_validation_set)
        print(error(y_pred,full_validation_labels))

        prediction = rf_classifier.predict(test_Set)
        df = pd.DataFrame(prediction)
        df.to_csv("rf_prediction.csv", index = None)

         ### some useful code
        #print(pd.crosstab(full_validation_labels, y_pred, rownames=['Classes'], colnames=['Predicted Classes']))
        #S.No,lat,lon,TMQ,U850,V850,UBOT,VBOT,QREFHT,PS,PSL,...
        #....,  T200,T500,PRECT,TS,TREFHT,Z1000,Z200,ZBOT,time,LABELS
        #print(list(zip(dataset.columns[0:20], rf_classifier.feature_importances_)))

        # Sving the model
        # joblib.dump(rf_classifier, 'randomforestmodel.pkl')

        # # load the model from disk to predict new dataSet
        # loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(X_test, Y_test)

        return rf_classifier, prediction




# NOTES

# My encoder
  # def one_hot(self, y):
  #       u_l=list(np.unique(y))
  #       encoded=np.zeros((len(y), len(u_l)))
  #       for i, c in enumerate(y):
  #           encoded[i][u_l.index(c)]=1
  #       return encoded


###  preporcessing Data
# transforming time in month for trainnignSet and testSet
# for i in range(trainig_x.shape[0]):
#     m = get_month_from_time(trainig_x.loc[i,'time'])
#     trainig_x.loc[i,'time'] = m

###  get_month_from_time(trainig_x,'time')
# mean_std  = meanStd(trainig_x)
# train_X = standardize_data(trainig_x, mean_std)
# train_X.to_csv('train_X_ready', index = False)

### ___PairPlots
# Var Names : S.No,lat,lon,TMQ,U850,V850,UBOT,VBOT,QREFHT,PS,PSL,...
#....,  T200,T500,PRECT,TS,TREFHT,Z1000,Z200,ZBOT,time,LABELS
# train_Set['LABELS'] = trainig_y  # adding labels to do PairPlot

# f = sns.pairplot(correctedTrainSet, hue = 'LABELS', diag_kind = 'kde',
#              plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
#              size = 4)# vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'],

# f = plt.figure()
# plt.ion
# plt.savefig('pairPlot.png', dpi=300)


###   Adding Bias to DataSet
# bias = np.ones((test_Set.shape[0]))
# test_Set.insert(0, 'bias', bias)
# test_Set.astype('float32')

###   Import Corrected  Data ''' To avoid correction avery time we call for debug or train a new model
# train_Set = pd.read_csv("train_X_ready.csv", index_col = None)
# trainig_y = pd.read_csv("trainig_y_ready.csv", index_col = None)
# test_Set = pd.read_csv("corrected_TestSet.csv", index_col = None)


###    Splitting trainig/Validation
# val_percent = int(np.abs((train_Set.shape[0])/3))
# val_index = np.random.randint(0,train_Set.shape[0],val_percent)
# validatoin_set = train_Set.drop(val_index)
# validatoin_labels = np.array(trainig_y.drop(val_index))
# trainig_y  = np.array(trainig_y)

### Trainning LR to find the better combination of hyperparameter

# lambd = [0.01,0.03,0.1,0.5,1]
# alpha = [0.01,0.03,0.1,0.5,1]
# results = pd.DataFrame({'TrainError' : [0],'alpha' : [0],'lambda' : [0],'valdationError' : [0]})

# for i in range(0,4):
#     j = 0
#     for j in range(0,4):
#         # trainig
#         lr=logisticRegression(train_Set, trainig_labels,lambd[i], alpha[j],1200)
#         theta_gd, loss, errors = lr.gradientDescent()
#         # predict and error calculation from validation set
#         predicter = formatPrediction(validatoin_set, theta_gd)
#         z_predict = predicter.hypotesis(validatoin_set, theta_gd)
          #predictedVal = predicter.prediction(z_predict,validatoin_labels)
#         predictedValidation = np.array(predictedVal['LABELS'])
#         validationError = lr.error(predictedValidation, validatoin_labels)
#         results = results.append({'TrainError' : errors[-1],'alpha': alpha[i],'lambda': lambd[j],'valdationError' : validationError}, ignore_index=True)
#         print('_test error', errors[-1], '__ alpha:', alpha[i], '__lambda:', lambd[j],'_valdation error:', validationError)


# results.to_csv('Summary_LR.csv', index = False)

# df['label'].value_counts()
