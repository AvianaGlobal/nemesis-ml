# coding: utf-8

import pandas as pd
import os


def flag_create(data, flag_name, condition, count):
    if count == 1:
        data[flag_name] = 0
    data.iloc[data.query(condition).index,len(data.columns)-1] = count
    print(data.head())
    return data

def flag_create_main(data):
    loop = 'Y'
    i = 1
    while loop == 'Y':
        flag_name = input('Enter the column name of the flag you want to create: ')
        condition = input('Enter the condition for the flag: ')
        if flag_name not in data.columns:
            i = 1
        try:
            data = flag_create(data, flag_name, condition, i)
            data.to_csv('Backup.csv')
            i = i + 1
            loop = input('Do you have another condition or want to create another flag? (Y/N) ')
            if loop == 'N':
                print(' ')
                print('Finished!')

        except:
            print('Invalid condition')

    os.remove("Backup.csv")
    return data






