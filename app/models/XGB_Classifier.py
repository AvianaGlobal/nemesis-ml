import pickle

import pandas as pd
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold



# import dataset
def main():
    data_file_name = input('Data file name: ')
    target = input('What is the column name of the target: : ')
    while True:
        tune = input('Do you want to tune model parameters? Y/N: ')
        if tune == 'Y' or tune == 'N':
            break
    df = pd.read_csv(data_file_name + '.csv')
    file = open(data_file_name + '_XGB_Classifier_report.txt', 'w')
    XGB_Classifier(file, df, target, data_file_name, tune)
    file.close()


def XGB_Classifier(file, train, test, target, data_file_name, tune):

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

    if tune == 'Y':
        # A parameter grid for XGBoost
        params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 2, 5],
            'subsample': [0.6, 1.0],
            'colsample_bytree': [0.6, 1.0],
            'max_depth': [3, 5],
            'n_estimators': [100, 300, 500]
        }

        # Instantiate and train the model

        model = XGBClassifier()

        folds = 3
        param_comb = 5

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

        random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                           n_jobs=4, cv=skf.split(X_train, y_train), verbose=3, random_state=0)

        # Here we go
        random_search.fit(X_train, y_train)

        print('\n All results:')
        print(random_search.cv_results_)
        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
        print(random_search.best_score_ * 2 - 1)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)
        results = pd.DataFrame(random_search.cv_results_)
        # save the model to disk
        results.to_csv('classifier-random-grid-search-xgb.csv', index=False)

        model = XGBClassifier(max_depth=random_search.best_params_['max_depth'], \
                              learning_rate=0.1, \
                              n_estimators=random_search.best_params_['n_estimators'], \
                              min_child_weight=random_search.best_params_['min_child_weight'], \
                              gamma=random_search.best_params_['gamma'], \
                              subsample=random_search.best_params_['subsample'], \
                              colsample_bytree=random_search.best_params_['colsample_bytree'], \
                              silent=True, objective='reg:linear')

    else:
        model = XGBClassifier()

    model.fit(X_train, y_train)

    # save the model to disk
    filename = data_file_name + '_XGBRegressor.sav'
    pickle.dump(model, open(filename, 'wb'))

    # prediction
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_test)

    # print the confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    file.write('2. Test set performance ')
    file.write('\n' + 'The confusion matrix is:\n' + str(cnf_matrix) + '\n')

    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    file.write("Accuracy: " + str(accuracy * 100.0))

    print('The model is', filename)


if __name__ == '__main__':
    main()
