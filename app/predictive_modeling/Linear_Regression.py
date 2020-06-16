import pickle

import pandas as pd
import statsmodels.formula.api as stfa
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# import dataset
def main():
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../sample_data/processed/' + data_file_name + '.csv')
    df_type = pd.read_csv('../../sample_data/processed/' + data_type_file_name + '.csv')
    file = open('../../model_results/build_models/' + data_file_name + '_linear_regression_report.txt', 'w')
    Linear_Regression(file, df, df_type, data_file_name)
    file.close()


def Linear_Regression(file, dataset, dataset_type, data_file_name, test_size=0.2):
    file.write('The current dataset has' + str(dataset.shape[1]) + ' columns and ' + str(dataset.shape[0]) + ' rows')
    file.write('Column names are ' + str(dataset.columns) + '\n')

    # funtion to get column type
    def column_type(column_name, df_type):
        return df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0]

    # Function to get target variable
    def get_target(df, df_type):
        for c in df:
            if column_type(c, df_type) == 'Flag_Continuous' or column_type(c, df_type) == 'Flag_Categorical':
                return c

    y = get_target(dataset, dataset_type)

    # seperate train set (80%) and test set (20%)
    d_train, d_test = train_test_split(dataset, test_size=test_size, random_state=0)
    dx_train = d_train.drop([y], axis=1)
    dy_train = d_train[[y]]
    dx_test = d_test.drop([y], axis=1)
    dy_test = d_test[[y]]

    # create a list to store simple LP model train and test error
    column_list = dx_train.columns.values.tolist()

    # get formula
    formula_list = y + '~'
    for column in column_list:
        formula_list = formula_list + ' + ' + column

    # fit simple LP model
    SLP = stfa.ols(formula=formula_list, data=d_train).fit()

    # save the model to disk
    filename = '../../predictive_modeling/' + data_file_name + '_LinearRegressor.sav'
    pickle.dump(SLP, open(filename, 'wb'))

    # prediction
    loaded_model = pickle.load(open(filename, 'rb'))
    ytest_pred1 = loaded_model.predict(dx_test)
    test_error = mean_squared_error(dy_test, ytest_pred1, sample_weight=None, multioutput='uniform_average')
    SLP_sum = SLP.summary()
    file.write(str(SLP_sum) + '\n')
    file.write('test error is ' + str(test_error))
    file.write('\nR Square is ' + str(r2_score(dy_test, ytest_pred1)))


if __name__ == '__main__':
    main()
