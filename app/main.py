import prediction, encoding, model_building

import pandas as pd

from data_cleaning import data_prep

while True:
    while True:
        datafile = input("Please enter the name of the csv data file: ")
        try:
            data = pd.read_csv(datafile + '.csv')
            break
        except:
            print('File does not exist!')
    data = data_prep(data)

    datafile = input("Please enter the name of the csv data file you want to save the processed data to: ")
    data.to_csv(datafile + '.csv', index = False)
    data_bool = input('Do you need to perform additional data prep? Y/N: ')

    if data_bool.upper() != 'Y':
        break

model_bool = input('Do you want to build a supervised machine learning model with this data? Y/N: ')
if model_bool.upper()=='Y':
    target_bool = input('Do you want to build a model (1), or to make predictions with an unlabeled data (2)? 1/2: ')
    while target_bool != '1' and target_bool != '2':
        target_bool = input('Do you want to build a model (1), or to make predictions with an unlabeled data (2)? 1/2: ')
    if target_bool == '1':
        train,test = encoding.encoding(data, True)
        model_building.model_building(train, test)
    elif target_bool == '2':
        data = encoding.encoding(data, False)
        data.to_csv(datafile +'_encoded.csv', index=False)
        prediction = prediction.prediction(data)
        prediction.to_csv(datafile + '_prediction.csv', index = False)


else:
    print('Finished!')

