import pickle

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../sample_data/processed/' + data_file_name + '.csv')
    df_type = pd.read_csv('../../sample_data/processed/' + data_type_file_name + '.csv')
    file = open('../../model_results/build_models/' + data_file_name + '_XGB_Classifier_report.txt', 'w')
    XGB_Classifier(file, df, df_type, data_file_name)
    file.close()


def XGB_Classifier(file, dataset, dataset_type, data_file_name, test_size=0.2):
    # Function to get target variable
    def get_target(df, df_type):
        for c in df:
            if column_type(c, df_type) == 'Flag_Continuous' or column_type(c, df_type) == 'Flag_Categorical':
                return c

    # funtion to get column type
    def column_type(column_name, df_type):
        return df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0]

    file.write('1. The current dataset has' + str(dataset.shape[1]) + ' columns and ' + str(dataset.shape[0]) + ' rows')
    file.write('\n' + 'Column names are ' + str(list(dataset.columns)) + '\n')
    file.write('\n' + 'The first 10 rows of this dataset: ' + '\n' + '\n' + str(dataset.head(10)) + '\n' + '\n')

    # split sample_data into X and Y
    target_name = get_target(dataset, dataset_type)
    X = dataset.drop(columns=target_name)
    Y = dataset[target_name]

    # split sample_data into train and test sets
    seed = 0
    test_size = test_size
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training sample_data
    model = XGBClassifier()
    model.fit(x_train, y_train)
    # save the model to disk
    filename = '../../predictive_modeling/' + data_file_name + '_XGBClassifier.sav'
    pickle.dump(model, open(filename, 'wb'))

    # important feature_engineering
    xgb.plot_importance(model)
    plt.show()

    # prediction
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(x_test)

    # print the confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    file.write('2. Test set performance ')
    file.write('\n' + 'The confusion matrix is:\n' + str(cnf_matrix) + '\n')

    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    file.write("Accuracy: " + str(accuracy * 100.0))


if __name__ == '__main__':
    main()
