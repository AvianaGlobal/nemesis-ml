import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

def main():
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../data/processed/'+data_file_name+'.csv')
    df_type = pd.read_csv('../../data/processed/'+data_type_file_name+'.csv')
    Logistic_Regression (dataset, dataset_type)



def Logistic_Regression (dataset, dataset_type, Test_size= 0.2):    
    
    print('The current dataset has',dataset.shape[1],'columns and',dataset.shape[0],'rows')
    print('\nColumn names are',dataset.columns)
    print('')
    print('The first 10 rows of this dataset:')
    print(dataset.head(10))
    
    
    # Funtion to get column type
    def column_type(column_name,df_type):
        return (df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0])
    
    # Function to get target variable
    def get_target(df,df_type):
        for c in df:
            if (column_type(c,df_type) == 'Flag_Continuous' or column_type(c,df_type) == 'Flag_Categorical'):
                return(c)
            
    # Separate the Train and Test set.      
    target = get_target(dataset,dataset_type)
    X = dataset.drop([target],axis = 1)
    Y = dataset.loc[:,target]
    
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size = Test_size, random_state=0)
    
    # Fit the Logistic Regression using train set.
    Logist_R = LogisticRegression(random_state=0, penalty = 'l2',solver='liblinear').fit(x_train, y_train)
    
    # save the model to disk
    filename = '../../models/Logistic_Regressor.sav'
    pickle.dump(Logist_R, open(filename, 'wb'))

    # Use the model to get the predict values
    y_pred = Logist_R.predict(x_test)
    
    # Cross Validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(Logist_R, x_train, y_train, cv=10)
    print('')
    print('')
    print('The cross validation accuracy list is',scores,'\n')
    print("The average is Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('')
    
    # Confusion metrix
    Logist_R_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    sns.heatmap(pd.DataFrame(Logist_R_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print("The Accuracy of Confusion Matrix is :",round(metrics.accuracy_score(y_test, y_pred),3))
    if dataset[target].dtype == int: 
        print("Precision:",round(metrics.precision_score(y_test, y_pred),3))
        print("Recall:",round(metrics.recall_score(y_test, y_pred),3))
