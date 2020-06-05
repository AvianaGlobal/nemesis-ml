# coding: utf-8

import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,  # training categorical feature as a pd.Series
                  tst_series=None,  # test categorical feature as a pd.Series
                  target=None,  # target data as a pd.Series
                  min_samples_leaf=1,  # minimum samples to take category average into account
                  smoothing=1,  # smoothing effect to balance categorical average vs prior
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    print(temp)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    print('average')
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def target_encode_main(data, target_col):
    # split train, test data
    data = data.dropna()
    train_test_series = input('Enter the column that you want to encode: ')
    size = float(input('Enter the training size: '))

    train_data, test_data, train_x, test_x, train_y, test_y = \
        train_test_split(data, data[train_test_series], data[target_col], test_size=size)

    temp_train, temp_test = target_encode(train_x, test_x, train_y)

    train_data[train_test_series + '_encoded'] = temp_train
    train_data = train_data.drop(columns=[train_test_series])

    test_data[train_test_series + '_encoded'] = temp_test
    test_data = test_data.drop(columns=[train_test_series])

    return train_data, test_data


def dummy(data, target_col):
    size = float(input('Enter the training size: '))

    train_data, test_data = train_test_split(data, test_size=size)

    train_data = pd.get_dummies(train_data, columns=[target_col], drop_first=False)
    test_data = pd.get_dummies(test_data, columns=[target_col], drop_first=False)

    return train_data, test_data


def cat_to_num(data):
    target_col = input('please enter your target column: ')

    if data[target_col].nunique() < 5:
        train_data, test_data = dummy(data, target_col)
    else:
        train_data, test_data = target_encode_main(data, target_col)

    return train_data, test_data


# test dataset
# df = pd.read_csv('card transactions_edited.csv')
# train, test = target_encode_main(df)

