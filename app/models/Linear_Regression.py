import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# import dataset
def main(data):
    data_file_name = input("What's the name of this data set:")
    target = input('What is the column name of the target: : ')
    file = open(data_file_name + '_Linear_regression_report.txt', 'w')
    Linear_Regression(file, data, target, data_file_name)
    file.close()


def Linear_Regression(file, train, test, target, data_file_name):
    file.write('1. The current dataset has ' + str(train.shape[1]) + ' columns. The training set has ' + str(
        train.shape[0]) + ' rows and the testing set has ' + str(test.shape[0]) + 'rows\n')
    file.write('\n' + 'Column names are  :' + str(list(train.columns)) + '\n')
    file.write('\n' + 'The first 10 rows of this dataset: ' + '\n' + '\n' + str(train.head(10)) + '\n' + '\n')
    y = target

    # seperate train set and test set
    X_train = train.drop(y, axis=1).values
    y_train = train[y].values
    X_test = test.drop(y, axis=1).values
    y_test = test[y].values

    # model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)  # training the algorithm
    regressor.feature_names = list(X_train.columns.values)

    # save the model to disk
    filename = data_file_name + '_LinearRegressor.sav'
    pickle.dump(regressor, open(filename, 'wb'))

    # prediction
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_test)

    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    file.write('\n' + 'The mean squared error is :' + str(RMSE) + '\n')

    print('The model is', filename)


if __name__ == '__main__':
    main()