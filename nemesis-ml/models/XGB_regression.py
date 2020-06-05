import pickle
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost.sklearn import XGBRegressor

warnings.filterwarnings("ignore")


# import dataset
def main():
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../data/processed/' + data_file_name + '.csv')
    df_type = pd.read_csv('../../data/processed/' + data_type_file_name + '.csv')
    file = open('../../reports/build_models/' + data_file_name + '_XGB_regression_report.txt', 'w')
    XGB_Regression(file, df, df_type, data_file_name)
    file.close()


def XGB_Regression(file, df, df_type, data_file_name):
    file.write('The current dataset has' + str(df.shape[1]) + ' columns and ' + str(df.shape[0]) + ' rows')
    file.write('Column names are ' + str(df.columns) + '\n')

    # funtion to get column type
    def column_type(column_name, df_type):
        return df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0]

    # Function to get target variable
    def get_target(df, df_type):
        for c in df:
            if column_type(c, df_type) == 'Flag_Continuous' or column_type(c, df_type) == 'Flag_Categorical':
                return c

    target_name = get_target(df, df_type)
    X = df.drop(columns=target_name)
    y = df[target_name]

    # seperate train set (80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Instantiate and train the model
    model = XGBRegressor(max_depth=1, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear')
    model.fit(X_train, y_train)

    # save the model to disk
    filename = '../../models/' + data_file_name + '_XGBRegressor.sav'
    pickle.dump(model, open(filename, 'wb'))

    # prediction
    loaded_model = pickle.load(open(filename, 'rb'))
    pred_test = loaded_model.predict(X_test)

    # important features
    plot_importance(loaded_model)
    plt.show()

    file.write('Test MSE: ' + str(mean_squared_error(y_test, pred_test)))
    file.write('R Square: ' + str(r2_score(y_test, pred_test)))


if __name__ == '__main__':
    main()
