# coding: utf-8
import pickle
from models import Linear_Regression, Logistic_Regression, XGB_regression, XGB_Classifier



def prediction(data):

    filename = input("What the model's filename? (.sav)")

    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(data)
    data['prediction'] = y_pred

    return data
