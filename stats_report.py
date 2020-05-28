# coding: utf-8

import pandas as pd 
from pandas.api.types import is_numeric_dtype

def stats_report(filepath,target_col,groupby_col):
    
    # read data
    df = pd.read_csv(filepath)
    if is_numeric_dtype(df[target_col]) == True:
        redo = 'Y'
        while redo == 'Y':
            colmax = df[target_col].groupby(df[groupby_col]).max()
            colmin = df[target_col].groupby(df[groupby_col]).min()
            colmean = df[target_col].groupby(df[groupby_col]).mean()
            colstd = df[target_col].groupby(df[groupby_col]).std()
            colsum = df[target_col].groupby(df[groupby_col]).sum()
            report = pd.DataFrame([colmean,colmin,colmax,colstd,colsum], index = ['Mean','Min','Max','Standard Deviation','Sum'])
            print(report)
            redo = input('Wanna a new report? (Y/N): ')
            if redo.upper() == 'Y':
                target_col = input('Enter a new target column: ')
                
                if is_numeric_dtype(df[target_col]) == True:
                    groupby_col = input('Enter a new groupby column: ')
                else:
                    print(str(target_col) + ' is not numeric')
                    break
                    
            else:
                print('Finish!')
                break
    else:
        print(str(target_col) + ' is not numeric')
        
    return None

