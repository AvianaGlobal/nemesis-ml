# coding: utf-8

from data_prep import flag_create, data_cleaning, import_filter, bin_insert, stats_report
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype
import warnings
warnings.filterwarnings('ignore')


def read_modification():
    modification = input('Do you want to make another change (Y/N) ')
    while True:
        if modification.upper() == 'Y' or modification.upper() == 'N':
            break
        modification = input('Invalid input. Do you want to make another change (Y/N) ')

    return modification.upper()


def data_prep(data):
    print('1 - change data type')
    print('2 - filter data')
    print('3 - fill NAs')
    print('4 - create flags')
    print('5 - create bins')
    print('6 - insert stats columns\n')
    operation = input('Do you want to make any change above to your data: (1/2/3/4/5/6/N) ')
    
    if operation != 'N':
        while True:

            if operation == '1':
                modification = 'Y'
                for col in data:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype('category')

                while modification.upper() == 'Y':
                    print(data.dtypes)
                    column = input('Enter the column name: ')
                    col_type = input('Enter the column type (e.g. int64, float64, category, datetime) ')
                    if col_type != 'datetime':
                        try:
                            data[column] = data[column].astype(col_type)
                            print(data.dtypes)
                            modification = read_modification()
                        except KeyError:
                            print('Column ' + str(column) + ' is not defined')
                            modification = read_modification()
                        except:
                            print('You cannot assign ' + str(col_type) + ' type to ' + str(column) + ' column')
                            modification = read_modification()
                    else:
                        try:  # when pandas can recognize the date format
                            data[column] = pd.to_datetime(data[column])
                            modification = read_modification()
                        except:  # when pandas cannot recognize the date format
                            date_format = input('Please enter the format of your data: i.e. %y/%m/%d: ')
                            to_datetime(data, column, date_format)
                            modification = read_modification()

            if operation == '2':

                try:
                    print('Here are the columns from your dataset: \n')
                    print(data.dtypes)
                    criterion = input('Please enter your filter: ')
                    data = import_filter.data_filter(data, criterion)
                except:
                    print('Invalid filter')

            if operation == '3':
                # identify numerical and categorical columns
                for col in data:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype('category')

                print(data.dtypes)
                modification = read_modification()

                while modification.upper() == 'Y':
                    column = input('Enter the column name: ')
                    col_type = input('Enter the column type (e.g. int64, float64, category, datetime) ')
                    if col_type != 'datetime':
                        try:
                            data[column] = data[column].astype(col_type)
                            print(data.dtypes)
                            modification = read_modification()
                        except KeyError:
                            print('Column ' + str(column) + ' is not defined')
                            modification = read_modification()
                        except:
                            print('You cannot assign ' + str(col_type) + ' type to ' + str(column) + ' column')
                            modification = read_modification()
                    else:
                        try:  # when pandas can recognize the date format
                            data[column] = pd.to_datetime(data[column])
                        except:  # when pandas cannot recognize the date format
                            date_format = input('Please enter the format of your data: i.e. %y/%m/%d: ')
                            to_datetime(data, column, date_format)

                while True:
                    print('Here are the columns from your dataset: \n')
                    print(data.columns.to_list())
                    print(data.dtypes)
                    print('Columns contains NAs' + str(data_cleaning.get_cols_with_NAs(data)))

                    target_col = input('please enter your target column: ')
                    groupby_col = input('Please enter your groupby column: ')
                    if target_col in data.columns and groupby_col in data.columns:
                        break
                    else:
                        print('Invalid target column or groupby column')
                try:
                    data = data_cleaning.clean_data_main(data, target_col, groupby_col)
                except:
                    print('There is an error when cleaning the data')

            if operation == '4':
                print('Here are the columns from your dataset: \n')
                print(data.dtypes)
                data = flag_create.flag_create_main(data)

            if operation == '5':
                while True:
                    print('Here are the columns from your dataset: \n')
                    print(data.dtypes)
                    target_col = input('please enter your target column: ')
                    groupby_col = input('Please enter your groupby column: ')
                    if target_col in data.columns and groupby_col in data.columns:
                        break
                    else:
                        print('Invalid target column or groupby column')
                data = bin_insert.bin_insert(data,target_col,groupby_col)

            if operation == '6':
                while True:
                    print('Here are the columns from your dataset: \n')
                    print(data.dtypes)
                    target_col = input('please enter your target column: ')
                    groupby_col = input('Please enter your groupby column: ')
                    if target_col in data.columns and groupby_col in data.columns:
                        break
                    else:
                        print('Invalid target column or groupby column')
                data = stats_report.stats_insert(data,target_col,groupby_col)


            print('1 - change data type')
            print('2 - filter data')
            print('3 - fill NAs')
            print('4 - create flags')
            print('5 - create bins')
            print('6 - insert stats columns\n')

            operation = input('Do you want to make another change? : (1/2/3/4/5/6/N) ')
            
            if operation.upper() == 'N':
                print('Finished! \n')
                print(data)
                break
                
    else:
        print('Finished! \n')
        print(data)
        os.remove("Backup.csv")
    return data
    
