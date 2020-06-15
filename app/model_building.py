# coding: utf-8

from models import Linear_Regression, Logistic_Regression, XGB_regression, XGB_Classifier


def model_building(train, test):
    print('1 - Classification')
    print('2 - Regression\n')
    operation = input('Is the prediction a classification or a regression problem: (1/2/Quit) ')

    if operation != 'Quit':
        data_file_name = input("What's the name of this data set:")
        target = input('What is the column name of the target column: : ')
        if operation == '1':
            model = input('Do you want to use a linear model (1) or a tree model (2): (1/2/Quit) ')
            if model == 'Quit':
                print('Finished! \n')

            elif model == '1':
                file = open(data_file_name + '_Logistic_regression_report.txt', 'w')
                Logistic_Regression.Logistic_Regression(file, train, test, target, data_file_name)
                file.close()
            elif model == '2':
                while True:
                    tune = input('Do you want to tune model parameters? Y/N: ')
                    if tune == 'Y' or tune == 'N':
                        break
                file = open(data_file_name + '_XGB_Classifier_report.txt', 'w')
                XGB_Classifier.XGB_Classifier(file, train, test, target, data_file_name, tune)
                file.close()

        if operation == '2':
            model = input('Do you want to use a linear model (1) or a tree model (2): (1/2/Quit) ')
            if model == 'Quit':
                print('Finished! \n')

            elif model == '1':
                file = open(data_file_name + '_Linear_regression_report.txt', 'w')
                Linear_Regression.Linear_Regression(file, train, test, target, data_file_name)
                file.close()
            elif model == '2':
                while True:
                    tune = input('Do you want to tune model parameters? Y/N: ')
                    if tune == 'Y' or tune == 'N':
                        break
                file = open(data_file_name + '_XGB_Regression_report.txt', 'w')
                XGB_regression.XGB_Regression(file, train, test, target, data_file_name, tune)
                file.close()
    else:
        print('Finished! \n')

