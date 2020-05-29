import pickle
import warning
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBRegressor
from matplotlib import pyplot as plt
from xgboost import plot_importance


# import dataset
def main():
    data_file_name = input('Data file name: ')
    target = input('What is the column name of the target: : ')
    while True:
        tune = input('Do you want to tune model parameters? Y/N: ')
        if tune == 'Y' or tune == 'N':
            break
    df = pd.read_csv(data_file_name + '.csv')
    file = open(data_file_name + '_XGB_regression_report.txt', 'w')
    XGB_Regression(file, df, target, data_file_name, tune)
    file.close()


def XGB_Regression(file, df, target, data_file_name, tune, test_size=0.2):
    def timer(start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    file.write('The current dataset has' + str(df.shape[1]) + ' columns and ' + str(df.shape[0]) + ' rows')
    file.write('Column names are ' + str(df.columns) + '\n')

    X = df.drop(columns=target)
    y = df[target]

    # seperate train set (80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

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

        model = XGBRegressor(max_depth=1, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear')

        folds = 3
        param_comb = 5

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

        random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb,
                                           scoring='neg_mean_squared_error', n_jobs=4, cv=skf.split(X_train, y_train),
                                           verbose=3, random_state=0)

        # Here we go
        start_time = timer(None)  # timing starts from this point for "start_time" variable
        random_search.fit(X_train, y_train)
        timer(start_time)

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
        results.to_csv('random-grid-search-xgb.csv', index=False)

        model = XGBRegressor(max_depth=random_search.best_params_['max_depth'], \
                             learning_rate=0.1, \
                             n_estimators=random_search.best_params_['n_estimators'], \
                             min_child_weight=random_search.best_params_['min_child_weight'], \
                             gamma=random_search.best_params_['gamma'], \
                             subsample=random_search.best_params_['subsample'], \
                             colsample_bytree=random_search.best_params_['colsample_bytree'], \
                             silent=True, objective='reg:linear')

    else:
        model = XGBRegressor(max_depth=1, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear')

    model.fit(X_train, y_train)

    # save the model to disk
    filename = data_file_name + '_XGBRegressor.sav'
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
