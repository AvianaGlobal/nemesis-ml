# import data which already build feature
#dataset = pd.read_csv('../../data/processed/kaggle_train_sample.csv')
#dataset_type = pd.read_csv('../../data/processed/kaggle_column_type.csv')

# import dataset
def main():
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../data/processed/'+data_file_name+'.csv')
    df_type = pd.read_csv('../../data/processed/'+data_type_file_name+'.csv')
    Linear_Regression(dataset, dataset_type)

def Linear_Regression(dataset,dataset_type,test_size=0.2):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import statsmodels.formula.api as stfa
    from sklearn.metrics import mean_squared_error
    
    print('The current dataset has',dataset.shape[1],'columns and',dataset.shape[0],'rows')
    print('Column names are',dataset.columns)
    print('')
    print('The first 10 rows of this dataset:')
    print(dataset.head(10))
    
    # funtion to get column type
    def column_type(column_name,df_type):
        return (df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0])
    #Function to get target variable
    def get_target(df,df_type):
        for c in df:
            if (column_type(c,df_type) == 'Flag_Continuous' or column_type(c,df_type) == 'Flag_Categorical'):
                return(c)
    y=get_target(dataset,dataset_type)
    # seperate train set (80%) and test set (20%)
    d_train, d_test = train_test_split(dataset, test_size=test_size, random_state=0)
    dx_train = d_train.drop([y],axis =1)
    dy_train = d_train[[y]]
    dx_test = d_test.drop([y],axis =1)
    dy_test = d_test[[y]]
    # create a list to store simple LP model train and test error
    column_list = dx_train.columns.values.tolist()
    # get formula
    formula_list = y + '~'
    for column in column_list:
        formula_list = formula_list + ' + ' + column
    # fit simple LP model
    SLP = stfa.ols(formula = formula_list, data = d_train).fit()
    ytest_pred1 = SLP.predict(dx_test)
    test_error = mean_squared_error(dy_test, ytest_pred1, sample_weight=None, multioutput='uniform_average')
    SLP_sum = SLP.summary()
    print(SLP_sum)
    print('')
    print(test_error)

Linear_Regression(dataset,dataset_type)

# save the model to disk
import pickle
filename = '../../models/LinearRegressor.sav'
pickle.dump(stfa.ols, open(filename, 'wb'))