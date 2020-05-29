# coding: utf-8

import pandas as pd

# Test file
df = pd.read_csv('C:\Users\ziruiw\Sample_Data\card transactions_edited.csv')
df.head()


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
        flag_name = input('Enter the flag_name: ')
        condition = input('Enter the condition for the flag: ')
        try:
            data = create_flag(data, flag_name, condition, i)
            i = i + 1
            loop = input('Do you have another condition or want to create another flag? (Y/N) ')
            if loop == 'N':
                print(' ')
                print('Finished!')
                
        except:
            print('Invalid condition')
            
    return data






