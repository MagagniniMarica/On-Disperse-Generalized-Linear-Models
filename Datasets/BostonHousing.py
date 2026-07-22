# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:37:47 2024

@author: Marica

Boston Dataset loading and preprocessing
"""

import pandas as pd
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split

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



    
# This function returs:
    # boston: the features DataDrame
    # target: the target Series
    # fetaures: the name of features Index
    # fetaures_type: the type of features Dict
def data(path, task):
    # data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # target = raw_df.values[1::2, 2]
    # features_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    # boston = pd.DataFrame(data, columns=features_names)
    
    
    full_path =path+'HousingData.csv'
    data = pd.read_csv(full_path)
    
    
    
    #imputation
    from sklearn.impute import SimpleImputer
    # Crea l'oggetto SimpleImputer per imputare con la media
    imputer = SimpleImputer(strategy='mean')  # 'mean', 'median', 'most_frequent'
    # Applica l'imputer sui dati
    boston = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    target = boston['MEDV']
    boston.drop('MEDV', axis=1, inplace=True)
    
    
    #Categorical variables
    boston['CHAS']=boston['CHAS'].astype('category') 
    # boston = boston.drop(['CHAS'], axis=1)
    
    #Target

    target = pd.Series(target)
    if task == 'classification' :
        target= target.apply(lambda x: 0 if x<=22 else 1) # for classification task
    
 
    features=boston.columns              # names of columns vector (Index)
    features_type=feature_type(boston)   # type of each feature vector (Dict)

    
    
    # # Split into training and test sets
    # boston_train, boston_test, target_train, target_test = train_test_split(boston, target, test_size=0.2, random_state=42)

    
    #Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    boston_scaled =pd.DataFrame( min_max_scaler.fit_transform(boston),  columns= features)


    
    
    return boston_scaled, target, features, features_type
    # return boston_train_scaled, boston_test_scaled, target_train, target_test, features, features_type
    