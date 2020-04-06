# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 08:51:15 2020

@author: MY PC
"""

import pandas as pd
import matplotlib as mt
import numpy as np
#splitting the datasets
ds=pd.read_csv("50_Startups.csv")
x=ds.iloc[:, :-1].values
y=ds.iloc[:,4].values

#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb=LabelEncoder()
x[:, 3]=lb.fit_transform(x[:, 3])
one=OneHotEncoder(categorical_features= [3])
x=one.fit_transform(x).toarray()

#dummyvariable trap
x=x[:,1:]

#splitting dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#scaling of the multiple linear regression is auto-matic

#multiple linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#prediction
y_pred=reg.predict(x_test)

#backward elimination
import statsmodels.regression.linear_model as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    mdl = regressor_OLS.get_robustcov_results(cov_type='HAC',maxlags=1)
    print( mdl.summary())
    return x
 
SL = 0.05
x= np.append(arr= np.ones((50,1)).astype(int),values=x,axis=1)
X_opt = x[:, [0, 1, 2, 3, 4,5]]
X_Modeled = backwardElimination(X_opt, SL)