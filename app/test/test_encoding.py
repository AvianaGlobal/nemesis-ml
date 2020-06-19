from app import encoding
from sklearn.model_selection import train_test_split
import pandas as pd
###### change the 'path' variable in encoding.py to  path = encoded_col + '_lookup.csv'

data = pd.read_csv('card transactions_edited_with_NAs.csv')
train_data, test_data = train_test_split(data, test_size=0.3)

def test_dummy():
    # data = pd.read_csv('card transactions_edited_with_NAs.csv')
    # train_data, test_data = train_test_split(data, test_size=0.3)
    train_data_d, test_data_d = encoding.dummy('Transtype', train_data, test_data)
    assert train_data.shape[1] - 1 + data['Transtype'].nunique() == train_data_d.shape[1]
    assert test_data.shape[1] - 1 + data['Transtype'].nunique() == test_data_d.shape[1]
    assert test_data_d.shape[1] == train_data_d.shape[1]

def test_addnoise():
    data_noise = encoding.add_noise(data['Amount'], 1)
    assert len((data['Amount'] != data_noise).unique()) == 1
    assert (data['Amount'] != data_noise).unique()[0] == True

def test_target_encode_main():
    train_x, test_x = train_data['MerchState'], test_data['MerchState']
    train_y, test_y = train_data['Fraud'], test_data['Fraud']
    train_data_t, test_data_t = encoding.target_encode_main('MerchState', train_data, test_data, train_x, test_x,
                                               train_y)

    assert 'MerchState' not in train_data_t.columns
    assert 'MerchState' not in test_data_t.columns
    assert 'MerchState_encoded' in train_data_t.columns
    assert 'MerchState_encoded' in test_data_t.columns
    assert train_data_t['MerchState_encoded'].dtype == 'float64'
    assert test_data_t['MerchState_encoded'].dtype == 'float64'

# def test_encoding_pred():
#
#     data_ep = encoding.encoding(data, False, 'Transtype')
#     assert data_ep.shape[1] >= 0