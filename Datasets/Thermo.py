# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""

from ucimlrepo import fetch_ucirepo 
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
    # fetch dataset 
    # infrared_thermography_temperature = fetch_ucirepo(id=925) 
    
    # # data (as pandas dataframes) 
    # X = infrared_thermography_temperature.data.features 
    # y = infrared_thermography_temperature.data.targets 
      
    
    # # Target 
    # target  = y['aveOralF'] 
    # target2 =  y['aveOralM']
    
    # # metadata 
    # print(infrared_thermography_temperature.metadata) 
      
    # # variable information 
    # print(infrared_thermography_temperature.variables) 
        
    
    
    # Read from local path
    full_path_data =path+'ITTuci_data.csv'
    full_path_target =path+'ITTuci_target.csv'
    X = pd.read_csv(full_path_data)
    y = pd.read_csv(full_path_target)
    target = y['aveOralF']
    
    # #
    # # Dataset selection
    # #
    # data = data.drop(0 , axis = 0)
    # data.columns = data.iloc[0]
    # data = data.drop(1 , axis = 0)
    
    # # features
    # features = [
    # # "SubjectID",
    # # "aveOralF",
    # # "aveOralM",
    # "Gender",
    # "Age",
    # "Ethnicity",
    # "T_atm",
    # "Humidity",
    # "Distance",
    # "T_offset1",
    # "Max1R13_1",
    # "Max1L13_1",
    # "aveAllR13_1",
    # "aveAllL13_1",
    # "T_RC1",
    # "T_RC_Dry1",
    # "T_RC_Wet1",
    # "T_RC_Max1",
    # "T_LC1",
    # "T_LC_Dry1",
    # "T_LC_Wet1",
    # "T_LC_Max1",
    # "RCC1",
    # "LCC1",
    # "canthiMax1",
    # "canthi4Max1",
    # "T_FHCC1",
    # "T_FHRC1",
    # "T_FHLC1",
    # "T_FHBC1",
    # "T_FHTC1",
    # "T_FH_Max1",
    # "T_FHC_Max1",
    # "T_Max1",
    # "T_OR1",
    # "T_OR_Max1"
    # ]
    
    
    
      
    # check missing values
    if X.isnull().values.any():
        missing_col = X.isnull().sum()
        print(missing_col)
        columns = missing_col[missing_col >0]
        columns_j = columns.index
        print(columns)
        rows = X[X.isnull().any(axis=1)]
        rows_i = rows.index
        print(rows)
        missing = X[columns_j].iloc[rows_i]
        print('missing = ',missing)
        
        num_imputer = SimpleImputer(strategy='mean')
        X[[columns_j[0]]] = num_imputer.fit_transform(X[[columns_j[0]]])


    # Set categorical features
    cat = ["Gender",
    "Age",
    "Ethnicity"
    ]
    for j in X.columns:
        # if X[j].dtype == 'O':
        if j in cat:
            X[j] = X[j].astype('category')
        else:
            X[j] = X[j].astype('float64')
            
           
     
    # Encode categorical features (ONE HOT ENCODING or ordinal for binary)
    # Initialize the OneHotEncoder
    thermo = copy.deepcopy(X) 
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
            thermo = thermo.drop(j, axis=1)
            thermo = pd.concat([thermo, encoded_tranformed], axis=1)
            
            
            
            
    
    
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in thermo.columns:
        if thermo[j].dtype != 'category':
            # print(thermo[j].dtype)
            thermo.loc[:, j]= min_max_scaler.fit_transform(thermo[[j]])
    
    
    
    
    
    # Display info
    thermo.keys()
    thermo.info()
    thermo.describe()
    features=thermo.columns              # names of columns vector (Index)
    features_type=feature_type(thermo)   # type of each feature vector (Dict)
    
    return thermo, target, features, features_type



  

  

