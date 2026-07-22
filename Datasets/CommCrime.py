# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""

import pandas as pd
import numpy as np
from os.path import join
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import copy


# This function returns a dict, for each feature is associated 
# the type (categorical (also binary), integer or numerical (continuaos))
def feature_type(dataset):
        features=list(dataset.columns)
        feat_type =[]
        for x in dataset.dtypes:
            if x.name == 'category':
                feat_type.append('Categorical')
            elif x.name == 'int64':
                feat_type.append('Integer')
            else:
                feat_type.append('Numerical')                
        features_type = dict(zip(features, feat_type))
        return features_type
    
    
def data(path, task):
    
    
    
    # from ucimlrepo import fetch_ucirepo 
  
    # # fetch dataset 
    # communities_and_crime = fetch_ucirepo(id=211) 
      
    # # data (as pandas dataframes) 
    # X = communities_and_crime.data.features 
    # y = communities_and_crime.data.targets 
      
    # # metadata 
    # print(communities_and_crime.metadata) 
      
    # # variable information 
    # print(communities_and_crime.variables) 
        
    ###########################################
    
    # Read from local path
    X_path =path+'CommCrime_data.csv'
    y_path = path + 'CommCrime_target.csv'
    X = pd.read_csv(X_path,  na_values="?")
    y = pd.read_csv(y_path, na_values="?")
    
    
    target = y['violentPerPop']
    
    
    
    
    # check missing values
    if X.isnull().values.any():
        missing_col = X.isnull().sum()
        print("Missing per colonna:\n", missing_col)
    
        # 1. colonne con più di 50 missing → DROP
        cols_to_drop = missing_col[missing_col > 50].index
        print("\nColonne eliminate (>50 missing):\n", cols_to_drop)
    
        X = X.drop(columns=cols_to_drop)
    
        # 2. colonne con missing ≤ 50 → IMPUTAZIONE
        cols_to_impute = missing_col[(missing_col > 0) & (missing_col <= 50)].index
        print("\nColonne da imputare:\n", cols_to_impute)
    
        if len(cols_to_impute) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X[cols_to_impute] = num_imputer.fit_transform(X[cols_to_impute])
    
        # debug finale
        print("\nMissing dopo trattamento:\n", X.isnull().sum())
        
     
    X = X[target.notnull()].reset_index(drop=True)
    target = target[target.notnull()].reset_index(drop=True)
   
    print(X.shape)
    print(target.shape)
    X.head()
    X.info()    
    print(X.dtypes)
    

    # Set categorical features

    for j in X.columns:
        if X[j].dtype == 'O':
        
            X[j] = X[j].astype('category')
        else:
            X[j] = X[j].astype('float64')
            
           
     
    # Encode categorical features (ONE HOT ENCODING or ordinal for binary)
    # Initialize the OneHotEncoder
    crime = copy.deepcopy(X) 
    features_multicat = {}
    encoder_OH = preprocessing.OneHotEncoder(sparse_output=False)  # sparse=False returns a dense array
    encoder_O  = preprocessing.LabelBinarizer() # single column for binary variables
    for j in X.columns:
        if X[j].dtype == 'category':
            if X[j].nunique() == 2: # binary
                encoded = encoder_O.fit_transform(X[[j]])
                encoded_tranformed = pd.DataFrame(encoded, columns = [j], dtype='category')
            else:    # multiple labels
                encoded = encoder_OH.fit_transform(X[[j]])
                num_j =len(encoder_OH.get_feature_names_out([j])) 
                encoded_tranformed = pd.DataFrame(encoded, columns=[j+f'_{i+1}' for i in range(num_j)], dtype='category')
                features_multicat[j] = encoded_tranformed.columns
            crime = crime.drop(j, axis=1)
            crime = pd.concat([crime, encoded_tranformed], axis=1)
            
            
            
            
    
    
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in crime.columns:
        if crime[j].dtype != 'category':
            # print(thermo[j].dtype)
            crime.loc[:, j]= min_max_scaler.fit_transform(crime[[j]])
    
    
    
    
    
    # Display info
    crime.keys()
    crime.info()
    crime.describe()
    features=crime.columns              # names of columns vector (Index)
    features_type=feature_type(crime)   # type of each feature vector (Dict)
    
    return crime, target, features, features_type



  

  

