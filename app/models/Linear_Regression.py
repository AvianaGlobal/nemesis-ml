import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# import dataset
def main():
    data_file_name = input('Data file name: ')
    target_column = input('What is the column name of the target: ')
    df = pd.read_csv(data_file_name + '.csv')
    Linear_Regression(df, target_column)


def Linear_Regression(dataset, target_column, test_size=0.2):
    y = target_column

    # seperate train set and test set
    X = dataset.drop(y, axis=1).values
    y = dataset[y].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)  # training the algorithm

    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Predicted': y_pred})

    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(df)
    return (df, MSE, RMSE)


if __name__ == '__main__':
    main()