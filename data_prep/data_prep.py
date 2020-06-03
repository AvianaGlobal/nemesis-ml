# coding: utf-8

import flag_create, data_cleaning, import_filter, bin_insert, stats_report
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype
import warnings
warnings.filterwarnings('ignore')


def data_prep(data):
    print('1 - filter data')
    print('2 - clean data') 
    print('3 - create flags')
    print('4 - create bins')
    print('5 - insert stats columns\n')
    operation = input('Do you want to make any change above to your data: (1/2/3/4/5/N) ')
    
    if operation != 'N':
        while True:

            if operation == '1':
                try:
                    criterion = input('Please enter your filter: ')
                    data = import_filter.data_filter(data, criterion)
                except:
                    print('Invalid filter')

            if operation == '2':
                while True:
                    target_col = input('please enter your target column: ')
                    groupby_col = input('Please enter your groupby column: ')
                    if target_col in data.columns and groupby_col in data.columns:
                        break
                    else:
                        print('Invalid target column or groupby column')
                data = data_cleaning.clean_data_main(data, target_col, groupby_col)

            if operation == '3':

                data = flag_create.flag_create_main(data)

            if operation == '4':
                while True:
                    target_col = input('please enter your target column: ')
                    groupby_col = input('Please enter your groupby column: ')
                    if target_col in data.columns and groupby_col in data.columns:
                        break
                    else:
                        print('Invalid target column or groupby column')
                data = bin_insert.bin_insert(data,target_col,groupby_col)

            if operation == '5':
                while True:
                    target_col = input('please enter your target column: ')
                    groupby_col = input('Please enter your groupby column: ')
                    if target_col in data.columns and groupby_col in data.columns:
                        break
                    else:
                        print('Invalid target column or groupby column')
                data = stats_report.stats_insert(data,target_col,groupby_col)
            
            operation = input('Do you want to make another change? : (1/2/3/4/5/N) ')
            
            if operation == 'N':
                print('Finished! \n')
                print(data)
                break
                
    else:
        print('Finished! \n')
        print(data)
        
    return data
    




# Test file
# df = pd.read_csv('card transactions_edited_with_NAs.csv')
# data = data_prep(df)





