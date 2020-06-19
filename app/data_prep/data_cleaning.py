# coding: utf-8

import pandas as pd
import warnings
from sklearn import preprocessing

warnings.filterwarnings('ignore')


def get_cols_with_NAs(data):
    columns_with_NAs = []
    for columns in data:
        if data[columns].isnull().values.any() == True:
            columns_with_NAs.append(columns)
    if len(columns_with_NAs) != 0:
        return columns_with_NAs
    else:
        print('There is no missing value in the given dataset')


# return the percentage of NAs in the column
def NAs_ratio(data, column):  # input the whole column
    return data[column].isnull().sum() / data[column].shape[0]


def fill_NAs(data, target_col, groupby_col):

    # check if each gourp has more than 20 values

    # temp constains info about the number of NAs in each group
    temp = data[target_col].isna().groupby(data[groupby_col]).sum().reset_index()
    temp = temp[temp[target_col] != 0]

    # temptoal contains info about the number of non-NA values in each group
    temptotal = data[[groupby_col, target_col]].groupby([groupby_col]).count().reset_index()

    skipgroup = []

    for i in range(len(temp)):
        # if the group has less than 20 values, we do not fill the NAs automatically
        if temptotal[temptotal[groupby_col] == temp[groupby_col].iloc[i]].iloc[0, 1] <= 20:
            print('There are too few datapoints in ' + str(temp[groupby_col].iloc[i]) + ' group.')
            skipgroup.append(temp[groupby_col].iloc[i])
    print('\n')
    print('Filling NAs with mean value in each control group\n')

    # fill NAs for groups that has enough datapoints
    data.loc[~data[groupby_col].isin(skipgroup), target_col] = \
    data[~data[groupby_col].isin(skipgroup)].groupby(groupby_col)[target_col].apply(lambda x: x.fillna(x.mean()))

    # if groupby column contains NAs, fill target column with the mean value
    if data[groupby_col].isna().any() == True:
        data[target_col] = data[target_col].fillna(data[target_col].mean())

    print(data[0:10])
    data.to_csv('Backup.csv', index = False)

    return data, skipgroup


def fill_NAs_no_enough_data(data, target_col, skipgroup, method, fill_num=None):
    print('Successfully filled NAs except for groups: ' + str(skipgroup))
    # print('Do you want to fill those using mean, median, mode, linear, a specific value, or remove?')
    # method = input('Please choose a method (mean, median, mode, value, linear, remove): ')
    if method == 'mean':
        data[target_col] = data[target_col].fillna(data[target_col].mean())
    if method == 'median':
        data[target_col] = data[target_col].fillna(data[target_col].median())
    if method == 'mode':
        data[target_col] = data[target_col].fillna(data[target_col].mode())
    if method == 'value':
        #         fill_num = input('Value to use to replace NAs: ')
        data[target_col] = data[target_col].fillna(fill_num)
    if method == 'linear':
        data[target_col] = data[target_col].interpolate(method='polynomial', order=2)
    if method == 'remove':
        data.drop(data[data[groupby_col].isin(skipgroup)].index, inplace=True)

    print('Filled NAs successfully for column: ' + str(target_col) + '\n')
    print('Columns contains NAs: ' + str(get_cols_with_NAs(data)) + '\n')
    data.to_csv('Backup.csv', index = False)

    return data


def scaler(data):
    numlist = []
    for col in data:
        if hasattr(pd.Series(data[col]), 'cat') == False:
            numlist.append(col)
    scaler = preprocessing.StandardScaler().fit(data[numlist])
    data[numlist] = scaler.transform(data[numlist])
    print('Successfully scaled data! \n')
    print(data[0:10])
    data.to_csv('Backup.csv', index = False)
    return data


def to_datetime(dataset, datetime_col, date_format='%y-%m-%d'):
    try:
        dataset[datetime_col + '_date'] = pd.to_datetime(dataset[datetime_col], format=date_format)
    except:
        print('Sorry, this datetime format cannot be recognized')


def get_week_of_year(dataset, datetime_col):
    dataset[datetime_col + '_wk'] = [i.weekofyear for i in dataset[datetime_col]]


def get_weekday(dataset, datetime_col):
    dataset[datetime_col + '_wkd'] = [i.weekday() for i in dataset[datetime_col]]


def get_month(dataset, datetime_col):
    dataset[datetime_col + '_mth'] = [i.month for i in dataset[datetime_col]]


def clean_data_main(data, target_col, groupby_col):
    while True:

        if hasattr(pd.Series(data[groupby_col]), 'cat') == False:
            print('Invalid groupby column. Groupby column must be categorical')

        if hasattr(pd.Series(data[target_col]), 'cat') == True and hasattr(pd.Series(data[groupby_col]), 'cat') == True:
            print('Filling NAs in a categorical col')
            data[target_col] = data[target_col].fillna(
                data[target_col].groupby(data[groupby_col]).agg(pd.Series.mode).iloc[0])
            print('Filled NAs successfully for column: ' + str(target_col) + '\n')
            print('Columns contains NAs' + str(get_cols_with_NAs(data)))

            data.to_csv('Backup.csv', index = False)

        if hasattr(pd.Series(data[target_col]), 'cat') == False and hasattr(pd.Series(data[groupby_col]),
                                                                            'cat') == True:

            if NAs_ratio(data, target_col) > 0.2:
                data.drop(target_col, axis=1, inplace=True)
                print('Dropped column: ' + str(target_col))

            else:
                data, skipgroup = fill_NAs(data, target_col, groupby_col)
                if len(skipgroup) == 0:
                    print('All groups have enough datapoints \n')
                    print('Filled NAs successfully for column: ' + str(target_col) + '\n')
                    print('Columns contains NAs' + str(get_cols_with_NAs(data)))
                else:
                    method = input('Please choose a method (mean, median, mode, value, linear, remove): ')
                    if method == 'value':
                        fill_num = input('Value to use to replace NAs: ')
                        data = fill_NAs_no_enough_data(data, target_col, skipgroup, method, fill_num)
                    else:
                        data = fill_NAs_no_enough_data(data, target_col, skipgroup, method)

        fill = input('Fill another column? (Y/N): ')
        if fill == 'Y':
            target_col = input('Enter a the name of the column that has NAs: ')
            groupby_col = input('Enter a new groupby column: ')
        else:
            break

    # scale numerical data
    scale = input('Do you want to scale numerical data (Y/N): ')
    if scale.upper() == 'Y':
        scalewy = input('Do you want to scale all the numerical data including the target (Y) column? Y/N')
        if scalewy.upper() == 'Y':
            data = scaler(data)
        else:
            target_col = input('Enter your target (Y) column')
            df = data.drop(columns=target_col)
            df = scaler(df)
            data = pd.concat([data[target_col], df], axis=1)

    print(data.dtypes)
    print(data)

    print('\n')
    print('Finished!')

    data = data.reset_index(drop=True)

    # remove backup data
    import os
    os.remove("Backup.csv")

    return data