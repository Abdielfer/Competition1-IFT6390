import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib


## Helper functions 
def formating_prediction(predictions): 
        '''
        return de predicted classes from the hypotesis function result (sigmoid(W,X))
        @hypotesis : matrix of probablilities 
        '''
        y_hat = pd.DataFrame({'S.No' : [],'LABELS' : []}, dtype=np.int8) 
        for i in range(len(predictions)):
            y_hat.loc[i] = [i,predictions[i]]
       
        return pd.DataFrame(data = y_hat) 
# 
def error(y_hat, y):
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



### _____ Import Data ____ ###
##   Full dataset standardized for training
train_set = pd.read_csv("correctedTrainSet_1.csv",index_col = None)
train_Y = pd.read_csv("trainig_y_ready.csv", index_col = None)
test_Set = pd.read_csv("corrected_TestSet_1.csv", index_col = None) # TestSet to predict


### preprocessing   : prtial dataser with standardized time 
full_train_set = pd.read_csv("train_X_ready.csv",index_col = None)
full_test_set = pd.read_csv("test_X_ready.csv",index_col = None)

train_time = full_train_set['time']
test_time = full_test_set['time']
    # Replacing in partial sets
train_set['time'] = train_time
test_Set['time'] = test_time


## Splitting trainig/Validation
val_percent = int(np.abs((train_set.shape[0])/3))
val_index = np.random.randint(0,train_set.shape[0],val_percent)
full_validation_set = train_set.drop(val_index) ## Validation DATA
full_validation_labels = np.array(train_Y.drop(val_index)) # Validation LABELS
labels = np.array(train_Y).reshape(-1)


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 42)
classifier.fit(train_set, labels)

# Predicting the Test set results
y_pred = classifier.predict(full_validation_set)
print(error(y_pred,full_validation_labels))

prediction = classifier.predict(test_Set)
df = formating_prediction(prediction)
df.to_csv("rf_prediction.csv", index = None)


#print(pd.crosstab(full_validation_labels, y_pred, rownames=['Classes'], colnames=['Predicted Classes']))
 #S.No,lat,lon,TMQ,U850,V850,UBOT,VBOT,QREFHT,PS,PSL,...
#....,  T200,T500,PRECT,TS,TREFHT,Z1000,Z200,ZBOT,time,LABELS
#print(list(zip(dataset.columns[0:18], classifier.feature_importances_)))

# Sving the model 
# joblib.dump(classifier, 'randomforestmodel.pkl')

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

