# coding: utf-8
import pickle
from models import Linear_Regression, Logistic_Regression, XGB_regression, XGB_Classifier



def prediction(data):
    while True:
        filename = input("What the model's filename? (.sav) ")
        try:
            loaded_model = pickle.load(open(filename+'.sav', 'rb'))
            break
        except:
            tryagain = input('Model does not exist. Do you want to try Again? Y/N')
            if tryagain.upper() != 'Y':
                break


        f_names = loaded_model.feature_names
        y_pred = loaded_model.predict(data[f_names].values)
        data['prediction'] = y_pred
        break

        return data
