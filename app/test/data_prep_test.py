from ..data_prep import flag_create
import pandas as pd
import pytest



def test_flag_create():
    data = pd.read_csv('card transactions_edited_with_NAs.csv')
    data = flag_create.flag_create(data, 'GreatAmount', 'Amount > 1000', 1)

    assert 'GreatAmount' in data

# def test_AssertTrue():
#     assert False