# coding: utf-8

import pandas as pd 
from pandas.api.types import is_numeric_dtype
import os

def stats_insert(df,target_col,groupby_col):

    redo = 'Y'
    data = df
    while redo.upper() == 'Y':
        if is_numeric_dtype(df[target_col]) == True:
            isstats = input('Which stats do you want to add into your data?  (eg.Mean,Min,Max,Standard Deviation(stdv),Sum)')
            if isstats.upper() == 'MAX':
                stats = df[target_col].groupby(df[groupby_col]).max()
            elif isstats.upper() == 'MIN':
                stats = df[target_col].groupby(df[groupby_col]).min()
            elif isstats.upper() == 'MEAN':
                stats = df[target_col].groupby(df[groupby_col]).mean()
            elif isstats.upper() == 'STDV':
                stats = df[target_col].groupby(df[groupby_col]).std()
            elif isstats.upper() == 'SUM':
                stats = df[target_col].groupby(df[groupby_col]).sum()
            else:
                print('Sorry, not available')
                break
            df1 = stats.to_frame(name = "Groupby"+ str(groupby_col) +isstats.upper())
            data = pd.merge(data,df1,on=groupby_col)
            print(data)
            data.to_csv('Backup.csv')

            redo = input('Wanna add a new stat? (Y/N): ')
            correct_col = False
            while redo.upper() == 'Y' and correct_col == False:
                target_col = input('Enter a new target column: ')
                try:
                    if is_numeric_dtype(df[target_col]) == True:
                        groupby_col = input('Enter a new groupby column: ')
                        while groupby_col in df.columns:
                            correct_col = True
                            break
                        else:
                            print('Invalid groupby column')
                    else:
                        print('Not numeric')
                except:
                    print('Invalid target column')


        else:
            print('Not numeric')

    print('Finish!')
    os.remove("Backup.csv")

    return data


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

