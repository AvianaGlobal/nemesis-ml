import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def main(data):
    data_file_name = input("What's the name of this data set:")
    target = input('What is the column name of the target: : ')
    file = open(data_file_name + '_Logistic_regression_report.txt', 'w')
    Logistic_Regression(file, data, target, data_file_name)
    file.close()


def Logistic_Regression(file, train, test, target, data_file_name):
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

    # Fit the Logistic Regression using train set.
    Logist_R = LogisticRegression(random_state=0, penalty='l2', solver='liblinear').fit(X_train, y_train)
    colname = list(train.columns.values)
    colname.remove(target)
    Logist_R.feature_names = colname

    # save the model to disk
    filename = data_file_name + '_LogisticRegressor.sav'
    pickle.dump(Logist_R, open(filename, 'wb'))

    # prediction
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_test)

    # Cross Validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(Logist_R, X_train, y_train, cv=10)
    list_S = list(scores)
    Rounded_list = [round(elem, 4) for elem in list_S]
    Avg_S = round(scores.mean(), 4)
    file.write('\n' + '\n' + '2. Cross validation results' + '\n')
    file.write('\n' + 'The cross validation accuracy list is ' + str(Rounded_list) + '\n')
    file.write('\n' + 'The average Accuracy is:  ' + str(Avg_S) + '\n')

    # Confusion metrix
    Logist_R_matrix = metrics.confusion_matrix(y_test, y_pred)

    sns.heatmap(pd.DataFrame(Logist_R_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    df_confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
                                       index=[['actual', 'actual'], ['neg', 'pos']],
                                       columns=[['predicted', 'predicted'], ['neg', 'pos']])
    file.write('\n' + '\n' + '3. Test set performance' + '\n')
    file.write('\n' + "Test confusion matrix: " + '\n')
    file.write(str(df_confusion_matrix) + '\n')
    Avg_Accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)
    file.write('\n' + 'The accuracy of confusion matrix is :' + str(Avg_Accuracy) + '\n')

    if train[target].dtype == int:
        precision = round(metrics.precision_score(y_test, y_pred), 3)
        Recall = round(metrics.recall_score(y_test, y_pred), 3)
        file.write("Precision: " + str(precision) + '\n')
        file.write("Recall: " + str(Recall) + '\n')

    print('The model is', filename)

if __name__ == '__main__':
    main()

