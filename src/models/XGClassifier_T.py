import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

def main():
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../data/processed/'+data_file_name+'.csv')
    df_type = pd.read_csv('../../data/processed/'+data_type_file_name+'.csv')
    XGB_Classifier (dataset, dataset_type)

def XGB_Classifier(dataset, dataset_type, test_size = 0.2):
    
    #Function to get target variable
    def get_target(df,df_type):
        for c in df:
            if (column_type(c,df_type) == 'Flag_Continuous' or column_type(c,df_type) == 'Flag_Categorical'):
                return(c)
            
    # funtion to get column type
    def column_type(column_name,df_type):
        return (df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0])
    
    print('The current dataset has',dataset.shape[1],'columns and',dataset.shape[0],'rows\n')
    print('Column names are', dataset.columns, '\n')
    print('The first 10 rows of this dataset:')
    print(dataset.head(10))
    
    # split data into X and Y
    target_name = get_target(dataset, dataset_type)
    X = dataset.drop(columns = target_name)
    Y = dataset[target_name]
    
    # split data into train and test sets
    seed = 0
    test_size = test_size
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
    
    # fit model no training data
    model = XGBClassifier()
    model.fit(x_train, y_train)
    
    # important features
    xgb.plot_importance(model)
    plt.show()
    
    # make predictions for test data
    y_pred = model.predict(x_test)
    
    # print the confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('The confusion matrix is:\n', cnf_matrix, '\n')
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # save the model to disk
    filename = '../../models/XGBClassifier.sav'
    pickle.dump(xgb_classifier, open(filename, 'wb'))