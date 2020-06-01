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

    print(data[0:10])
    data.to_csv('Backup.csv')

    return data, skipgroup


def fill_NAs_no_enough_data(data, skipgroup, method, fill_num=None):
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
    data.to_csv('Backup.csv')

    return data


def scaler(data, scale):
    numlist = []
    for col in data:
        if hasattr(pd.Series(data[col]), 'cat') == False:
            numlist.append(col)
    if scale.upper() == 'Y':
        scaler = preprocessing.StandardScaler().fit(data[numlist])
        data[numlist] = scaler.transform(data[numlist])
        print('Successfully scaled data! \n')
        print(data[0:10])
        data.to_csv('Backup.csv')
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



def main(filepath, target_col, groupby_col):
    # read data
    data = pd.read_csv(filepath)

    # identify numerical and categorical columns
    for col in data:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')

    print(data.dtypes)
    modification = input('Do you want to make any change (Y/N) ')

    while modification.upper() == 'Y':
        column = input('Enter the column name: ')
        col_type = input('Enter the column type (e.g. int64, float64, category, datetime) ')
        if col_type != 'datetime':
            try:
                data[column] = data[column].astype(col_type)
                print(data.dtypes)
                modification = input('Do you want to make another change (Y/N) ')
            except KeyError:
                print('Column ' + str(column) + ' is not defined')
                modification = input('Do you want to make another change (Y/N) ')
            except:
                print('You cannot assign ' + str(col_type) + ' type to ' + str(column) + ' column')
                modification = input('Do you want to make another change (Y/N) ')
        else:
            try:  # when pandas can recognize the date format
                data[column] = pd.to_datetime(data[column])
            except:  # when pandas cannot recognize the date format
                date_format = input('Please enter the format of your data: i.e. %y/%m/%d: ')
                to_datetime(data, column, date_format)

    print(data[data.isna().any(axis=1)].head())
    drop_cat_NA = input('Drop all NAs in categorical columns? (Y/N): ')
    if drop_cat_NA == 'Y':
        print(data[data.isna().any(axis=1)].head())

    # drop NAs in categorical columns
    if drop_cat_NA.upper() == 'Y':
        for col in data:
            if hasattr(pd.Series(data[col]), 'cat') == True:
                data = data.dropna(subset=[col])

    if NAs_ratio(data, target_col) > 0.2:
        data.drop(target_col, axis=1, inplace=True)
        print('Dropped column: ' + str(target_col))

    else:
        fill = 'Y'
        while fill == 'Y':
            data, skipgroup = fill_NAs(data, target_col, groupby_col)
            if len(skipgroup) == 0:
                print('All groups have enough datapoints \n')
                print('Filled NAs successfully for column: ' + str(target_col) + '\n')
                print('Columns contains NAs' + str(get_cols_with_NAs(data)))
            else:
                method = input('Please choose a method (mean, median, mode, value, linear, remove): ')
                if method == 'value':
                    fill_num = input('Value to use to replace NAs: ')
                    data = fill_NAs_no_enough_data(data, skipgroup, method, fill_num)
                else:
                    data = fill_NAs_no_enough_data(data, skipgroup, method)
            fill = input('Fill another column? (Y/N): ')
            if fill == 'Y':
                target_col = input('Enter a new target column: ')
                groupby_col = input('Enter a new groupby column: ')
            else:
                break

        # scale numerical data
        scale = input('Do you want to scale numerical data (Y/N): ')
        data = scaler(data, scale)

        print('\n')
        print('Finished!')

        # remove backup data
        import os
        os.remove("Backup.csv")
        print('Backup file has been deleted')

        # Export clean data
        data.to_csv(str(filepath + '_cleaned'))

        return data