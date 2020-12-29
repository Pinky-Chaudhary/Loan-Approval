# -*- coding: utf-8 -*-
"""
## Importing Libraries
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

"""# Processing Data"""

df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv").drop(['Unnamed: 0'],axis = 1)

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')

"""# Preprocessing"""

def Transforming(df):
  df['Gender'] = df['Gender'].fillna( df['Gender'].dropna().mode().values[0] )
  df['Married'] = df['Married'].fillna( df['Married'].dropna().mode().values[0] )
  df['Dependents'] = df['Dependents'].fillna( df['Dependents'].dropna().mode().values[0] )
  df['Self_Employed'] = df['Self_Employed'].fillna( df['Self_Employed'].dropna().mode().values[0] )
  df['LoanAmount'] = df['LoanAmount'].fillna( df['LoanAmount'].dropna().mean() )
  df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna( df['Loan_Amount_Term'].dropna().mode().values[0] )
  df['Credit_History'] = df['Credit_History'].fillna( df['Credit_History'].dropna().mode().values[0] )
  df['Dependents'] = df['Dependents'].str.rstrip('+')
  df['Gender'] = df['Gender'].map({'Female':1,'Male':0}).astype(np.int)
  df['Married'] = df['Married'].map({'No':0, 'Yes':1}).astype(np.int)
  df['Education'] = df['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
  df['Self_Employed'] = df['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)
  df['Dependents'] = df['Dependents'].astype(np.int)
  df['Loan_ID'] =df['Loan_ID'].str.strip("LP").astype(np.int)  
  df = pd.get_dummies(df)
  return df

train = Transforming(df)
test = Transforming(test_data)
target = 'Loan_Status'

"""#XGB_Classifier"""

xgbc = XGBClassifier(learning_rate =0.5, n_estimators=11, max_depth=12, min_child_weight=0, gamma=6, subsample=0.7755102040816326,
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=2)

xgbc.fit(train[test.columns],train[target])
pred = xgbc.predict(test)
pd.DataFrame({'prediction':pred}).to_csv('final_prediction.csv',index = False)

"""# Pickling"""

import pickle

with open('model.pickle','wb') as file:
  pickle.dump(xgbc,file)

