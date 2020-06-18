from ..data_prep import flag_create, import_filter, data_cleaning
import pandas as pd
from pandas.api.types import is_numeric_dtype
import random
import os
import pytest


def fill_NAs_cat(data, target_col, groupby_col):
    # Fill categorical column
    data[target_col] = data[target_col].fillna(data[target_col].groupby(data[groupby_col]).agg(pd.Series.mode).iloc[0])
    return data

def bin_insert(df, target_col, groupby_col, nbin):

    df = df.sort_values(target_col)
    if is_numeric_dtype(df[target_col]) == True:
        ###overview###
        n = df[target_col].count()
        print('The variable ' + str(target_col) + ' have ' + str(n) + ' records')
        ###break down into bins###

        redo = 'Y'
        data = pd.DataFrame(columns=df.columns)
        while redo.upper() == 'Y':

            if nbin > 0:

                binrg = round(n / nbin)
                i = 1
                while i <= nbin - 1:
                    df1 = df[int((i - 1) * binrg):int(i * binrg)]
                    df1 = df1.assign(bin=int(i))
                    data = data.append(df1)
                    i = int(i + 1)
                df2 = df[int((nbin - 1) * binrg):n]
                df2 = df2.assign(bin=nbin)
                data = data.append(df2)
                print(data)
                data.to_csv('Backup.csv')

                print('Start grouping....')
                ###grouping###
                g = 1
                while g <= nbin:
                    count = data[data['bin'] == g].groupby(groupby_col).count()[['bin']]
                    count = count.rename(columns={'bin': 'bin_is_' + str(g) + 'within_group'})
                    data = pd.merge(data, count, on=groupby_col)
                    g = g + 1

                print("Here's the new data! \n")
                print(data)

                os.remove("Backup.csv")
                break

            else:
                print('Please enter a valid positive integer: ')
                redo = 'Y'


    else:
        print(str(target_col) + ' is not numeric.')

    return data


def stats_insert(df, target_col, groupby_col, isstats):
    redo = 'Y'
    data = df
    if is_numeric_dtype(df[target_col]) == True:
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
        df1 = stats.to_frame(name="Groupby" + str(groupby_col) + isstats.upper())
        data = pd.merge(data, df1, on=groupby_col)

    return data



# -------------------------------------------------------------------------------
# test modules
def test_flag_create():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    colnum = int(data.shape[1])
    data = flag_create.flag_create(data, 'GreatAmount', 'Amount > 1000', 1)

    assert {'GreatAmount'}.issubset(data.columns)
    assert int(data.shape[1]) == colnum + 1

def test_import_filter():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    data = import_filter.data_filter(data,'Amount > 1000')
    assert (data.Amount > 1000).all()

def test_fill_NAs():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    data = data.dropna(subset=['MerchState'])
    data, skipgroup = data_cleaning.fill_NAs(data, 'Amount', 'MerchState')
    assert data['Amount'].isna().any() == False


def test_fill_cat_NAs():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    data = fill_NAs_cat(data, 'MerchState', 'Merchnum')

    assert data['MerchState'].isna().any() == False

def test_bin_insert():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    colnum = int(data.shape[1])
    data = bin_insert(data, 'Amount', 'MerchState', 5)

    assert int(data.shape[1]) == colnum + 6
    assert int(data.bin.unique().shape[0]) == 5




def test_stats_insert():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    colnum = int(data.shape[1])
    data = stats_insert(data, 'Amount', 'MerchState', 'min')

    # generate a random number
    randnum = random.randint(1, 90000)
    randstate = data.iloc[randnum]['MerchState']

    # result calculated directly from the dataset
    actual = data['Amount'].groupby(data['MerchState']).min()[randstate]

    # result extracted from the generated col
    result = data[data.columns[-1]][randnum]

    assert actual == result
    assert int(data.shape[1]) == colnum + 1
