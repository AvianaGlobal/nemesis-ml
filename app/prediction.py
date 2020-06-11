# coding: utf-8
import pickle
from models import Linear_Regression, Logistic_Regression, XGB_regression, XGB_Classifier



def prediction(data):

    filename = input("What the model's filename? (.sav)")

    loaded_model = pickle.load(open(filename, 'rb'))
    f_names = loaded_model.feature_names
    y_pred = loaded_model.predict(data[f_names])
    data['prediction'] = y_pred

    return data
