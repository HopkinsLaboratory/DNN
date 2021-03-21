# Supervised ML: Random Forest
"""
Created on Mon Mar 1 13:44:50 2021
Author: WSH
"""

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from csv import writer

# Import the DMS dataset
data = pd.read_csv('DMS_CCS_ML_Dataset.csv')
data.head()

#store the data in the form of dependent and independent variables separately
X = data.iloc[:, 0:8].values
y = data.iloc[:, 9].values

ts = 0.4

with open('output.csv', 'a', newline='') as f_object:

    #create error curve
    while ts > 0.01:
        #split the dataset into training and test datasets. scikit-learn recommends ensuring stability wrt variance by
		#initializing the random_state on different iterations with different constants. They suggest random_state = 0 and 42.
		#random_state = 30, 66, and 99 were also tested.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts, random_state = 0)
        
        #create a Random Forest regressor object from Random Forest Regressor class. scikit-learn recommends ensuring stability wrt variance by
		#initializing the random_state on different iterations with different constants. They suggest random_state = 0 and 42.
		#random_state = 30, 66, and 99 were also tested.
        RFReg = RandomForestRegressor(n_estimators = 500, random_state = 0) #n-estimators is the number of trees.
        
        #fit the random forest regressor with training data represented by X_train and y_train
        RFReg.fit(X_train, y_train)
        
        #predicted CCS from test dataset wrt Random Forest Regression
        y_predict_rfr = RFReg.predict((X_test))
        y_train_pred = RFReg.predict((X_train))
              
        #Model evaluation using R-square from Random Forest Regression
        r_square = metrics.r2_score(y_test, y_predict_rfr)
        #print('R-square error associated with Random Forest Regression is:', r_square)
        
		#Model evaluation mean absolute error
        MAE = metrics.mean_absolute_error(y_test, y_predict_rfr)
        #print('MAE is:', MAE)
        
		#Calculate mean absolute percentage error.
        def MAPE(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
             
		#Model evaluation mean absolute percentage error
        mape = MAPE(y_test, y_predict_rfr)
        mape_std = MAPE_std(y_test, y_predict_rfr)
        
        #print('MAPE is:', mape)
    
        params = [ts, r_square, MAE, mape]
        
        writer_object = writer(f_object) 
        writer_object.writerow(params)
        ts = ts - 0.005		#step the train/test split value for the error curve calculation.

"""
#To write CCS targets and predictions.
a = np.array(y_train)
b = np.array(y_train_pred)
c = np.array(y_test)
d = np.array(y_predict_rfr)

df = pd.DataFrame({"y_train" : a, "y_train_pred" : b})
df.to_csv("train_predictions.csv", index=False)

df = pd.DataFrame({"y_test" : c, "y_predict_rfr" : d})
df.to_csv("test_predictions.csv", index=False)
"""

f_object.close() 
