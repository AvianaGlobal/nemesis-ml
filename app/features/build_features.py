import pandas as pd
import numpy as np
from scipy import stats
import sklearn
from sklearn import preprocessing
from CHAID import Tree
import re
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
# from time import time
#
# start = time.clock()
import datetime
starttime = datetime.datetime.now()


#function to get number of missing values in a column
def get_na_num(column): #input the whole column
    if column.dtype == np.object:
        return column.isnull().sum() + column[column == ''].count() + column[column == '?'].count()
    else:
        return column.isnull().sum()

#function to get number of valid values in a column
def get_valid_num(column): #input the whole column
    return len(column) - get_na_num(column)

#function to get minimum value in a column
def get_min(column): #input the whole column
    new_column = column.dropna()
    return min(column)

#function to get maximum value in a column
def get_max(column): #input the whole column
    new_column = column.dropna()
    return max(column)

#function to get mean in a column
def get_mean(column): #input the whole column
    new_column = column.dropna()
    return column.mean()

#function to get std in a column
def get_std(column): #input the whole column
    new_column = column.dropna()
    return column.std()

#function to get skewness in a column
def get_skew(column): #input the whole column
    new_column = column.dropna()
    return column.skew()

#function to get number of distict values in a column
def get_distinct_num(column): #input the whole column
    return len(column.unique().tolist())

#function to get count of each distict value in a column
def get_distinct_count(column): #input the whole column
    dis_count={}
    values = column.value_counts().to_list()       
    for i in range(get_distinct_num(column)):
        key = column.value_counts().index[i]
        value = values[i]
        dis_count[key] = value
    return dis_count

#function to get median in a column
import statistics
def get_median(column):
    new_column = column.dropna()
    return statistics.median(column)

#function to get mode and count for the mode in a column
def get_mode(column):
    return (column.mode()[0],column[column==column.mode()[0]].count())

#Function to get target variable
def get_target(df,df_type):
    for c in df:
        if (column_type(c,df_type) == 'Flag_Continuous' or column_type(c,df_type) == 'Flag_Categorical'):
            return(c)

# funtion to get column type
def column_type(column_name,df_type):
    return (df_type.loc[df_type['Variable'] == column_name, 'Type'].iloc[0])

#function to do basic variable screening and create basic statistical report
def Stats_Collection(file,df,df_type):
    for c in df:
        file.write('\n')
        #exclude Target 
        if (column_type(c,df_type) != 'Flag_Continuous' and column_type(c,df_type) != 'Flag_Categorical'):
            file.write('Variable name: '+c+'\n')

            #Basic variable screening
            if get_na_num(df[c])/len(df[c]) > 0.5:
                file.write('More 50% missing values, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Delete'):
                file.write('Column type is Delete, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Continuous') and (get_min(df[c]) == get_max(df[c])):
                file.write('All same value, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Ordinal' or column_type(c,df_type) == 'Nominal') and (get_mode(df[c])[1]/get_valid_num(df[c]) > 0.95):
                file.write('Mode contains more than 95% cases, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Nominal') and (get_distinct_num(df[c]) > 100):
                file.write('More than 100 categories, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue

            #Basic statistic report
            file.write('Variable type: '+column_type(c,df_type)+'\n')
            file.write('Number of missing values: '+str(get_na_num(df[c]))+'\n')
            file.write('Number of valid values: '+str(get_valid_num(df[c]))+'\n')
            if column_type(c,df_type) == 'Continuous':
                file.write('Minimum value: ' + str(get_min(df[c])) + '\n')
                file.write('Maximum value: ' + str(get_max(df[c])) + '\n')
                file.write('Mean: '+str(get_mean(df[c]))+'\n')
                file.write('Standard Deviation: '+str(get_std(df[c]))+'\n')
                file.write('Skewness: '+str(get_skew(df[c]))+'\n')
                file.write('Number of distinct values: '+str(get_distinct_num(df[c]))+'\n')
                file.write('Number of cases for each distinct value: \n')
                if get_distinct_num(df[c]) > 5:
                    file.write('Number of distict values is larger than 5. We stop updating the number of distinct values\n')
                else:
                    dict_count = get_distinct_count(df[c])
                    for k, v in dict_count.items():
                        file.write(str(k) + ' >>> '+ str(v) + '\n')
            else:
                file.write('Number of categories: '+str(get_distinct_num(df[c]))+'\n')
                file.write('The counts of each category:\n')
                if get_distinct_num(df[c]) > 5:
                    file.write('Number of distict categories is larger than 5. We stop updating the number of distinct values\n')
                else:
                    dict_count = get_distinct_count(df[c])
                    for k, v in dict_count.items():
                        file.write(str(k) + ' >>> '+ str(v) + '\n')
                file.write('Mode: '+str(get_mode(df[c])[0])+', Count: '+str(get_mode(df[c])[1])+'\n')               
    return(df,df_type)

# function to identify outliers in continuous variables
# returns a bool panda series, a lower cutoff value and an upper cutoff value(used in outlier handling)
def outlier_identification(column):
    ori_mean = get_mean(column.dropna())
    ori_std = get_std(column.dropna())
    N_i = []
    X_i = []
    M_i = []
    for i in range(-3, 5):
        lower = ori_mean + (i - 1) * ori_std if i != -3 else -float('inf')
        upper = ori_mean + i * ori_std if i != 4 else float('inf')
        temp1 = column[(column <= upper) & (column > lower)]
        N_i.append(len(temp1))
        X_i.append(get_mean(temp1))
        M_i.append(np.var(temp1) * len(temp1))
    l = -3
    r = 4
    p = 0
    while 1:
        if N_i[l + 3] <= N_i[r + 3]:
            p_current = N_i[l + 3] / get_valid_num(column)
            if p + p_current < 0.05:
                l = l + 1
                p = p + p_current
            else:
                break
        else:
            p_current = N_i[r + 3] / get_valid_num(column)
            if p + p_current < 0.05:
                r = r - 1
                p = p + p_current
            else:
                break
    lower = ori_mean + (l - 1) * ori_std if l != -3 else -float('inf')
    upper = ori_mean + r * ori_std if r != 4 else float('inf')
    temp1 = column[(column <= upper) & (column > lower)]
    x_robust = get_mean(temp1)
    M_robust = 0
    N_sum = 0
    for i in range(l, r + 1):
        A_i = M_i[i + 3] + N_i[i + 3] * (x_robust - X_i[i + 3])**2
        M_robust = M_robust + A_i
        N_sum = N_sum + N_i[i + 3]
    sd_robust = np.sqrt(M_robust / (N_sum - 1))
    result = (column - x_robust < -3 * sd_robust) | (column - x_robust > 3 * sd_robust)
    lower_cutoff_value = x_robust - 3 * sd_robust
    upper_cutoff_value = x_robust + 3 * sd_robust
    return(result, lower_cutoff_value, upper_cutoff_value)

# function to trim outliers to cutoff values
def outlier_trim(df,column):
    (flag, lower, upper) = outlier_identification(column)
    df.loc[flag, column.name] = pd.Series(map(lambda x : (lower if x <= lower else upper), column[flag]))


# function to fill missing value and update statistic
def fill_missing_value(mydata, column_type):
    # filling missing value and updata statistic
    # Get target variable name
    # target = get_target(mydata, column_type)
    # # if there are missing value in target variable, drop that rows
    # na_row = mydata[target][mydata[target].isnull().values == True].index.tolist()
    # mydata = mydata.drop(index=na_row)
    # # drop target column
    # mydata = mydata.drop(columns=target)
    # column_type_index = column_type['Type'][column_type['Type'] == 'Flag_Continuous'].index.tolist()
    # column_type = column_type.drop(column_type_index)
    column_list = mydata.columns.values.tolist()
    typelist = list(column_type.iloc[:, 1])
    i = 0
    for typ in typelist:
        column_data = mydata[column_list[i]].dropna()
        if (typ == 'Flag_Continuous' or typ == 'Flag_Categorical'):
            print('')
            print('Target do not need to fill missing value')
        elif (typ == 'Continuous'):
            mean_value = column_data.mean() # calculate mean
            mydata[column_list[i]] = mydata[column_list[i]].fillna(mean_value) # fill missing value with mean
            cont_sd = mydata[column_list[i]].std() # calculate standard deviation
            cont_skew = mydata[column_list[i]].skew() # calculate skewness
            """print('')
            print('Column:', column_list[i])
            print('Column type: continuous')
            print('Mean:', mean_value)
            print('Standard deviation:', cont_sd)
            print('Skewness:', cont_skew)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))"""
        elif (typ == 'Ordinal' and mydata[column_list[i]].dtype != 'object'):
            num_median = column_data.median() # calculate median 
            mydata[column_list[i]] = mydata[column_list[i]].fillna(num_median) # fill missing value with median
            count_median_num = mydata[column_list[i]][mydata[column_list[i]] == num_median].count() # count the the number of cases in the median category
            """print('')
            print('Column:', column_list[i])
            print('Column type: num_ordinal')
            print('Median:', num_median)
            print('The number of cases in the median category:', count_median_num)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))"""
        elif (typ == 'Ordinal' and mydata[column_list[i]].dtype == 'object'):
            mode_value = column_data.mode()[0] # calculate mode
            mydata[column_list[i]] = mydata[column_list[i]].fillna(mode_value) # fill missing valye with mode
            count_mode = mydata[column_list[i]][mydata[column_list[i]] == mode_value].count() # count the the number of cases in the modal category
            """print('')
            print('Column:', column_list[i])
            print('Column type: cat_ordinal')
            print('Mode:', mode_value)
            print('The number of cases in the modal category:', count_mode)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))"""
        else:
            mode_value = column_data.mode()[0] # calculate mode
            mydata[column_list[i]] = mydata[column_list[i]].fillna(mode_value) # fill missing valye with mode
            count_mode = mydata[column_list[i]][mydata[column_list[i]] == mode_value].count() # count the the number of cases in the modal category
            """print('')
            print('Column:', column_list[i])
            print('Column type: nominal')
            print('Mode:', mode_value)
            print('The number of cases in the modal category:', count_mode)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))"""
        i = i + 1

    return (mydata)

#function to do z-score transformation of a column
def zscore(df, column,df_type):
    if column_type(column,df_type) == 'Continuous':
        return(stats.zscore(df[column]))
    else:
        return(df[column])

#function to do min-max transformation of a column
def minmax(df, column,df_type):
   if column_type(column,df_type) == 'Continuous':
       return(preprocessing.minmax_scale(df[column], feature_range=(0, 100)))
   else:
       return(df[column])

#function to do Box-Cox transformation of continuous target
def boxcox(df, target, df_type):
    if column_type(target, df_type) == 'Flag_Continuous':
        c = min(df[target]) - 1
        target1 = df[target] - c
        target1 = stats.boxcox(target1)[0]
        return(pd.Series(stats.zscore(target1)))
    else:
        return(df[target])


# Function to sort the Series by value and then by index(Lexical order)
def sort_data(Series):
    return Series.iloc[np.lexsort([Series.index, Series.values])]


# Function to supervised merged categories in categorical variables
# df = dataset, Predictor_type = Nominal or Ordinal, dependent_variable_name = target name, indep_column_num = column index
def Supervised_Merged (file,df, Predictor_type, dependent_variable_name, indep_column_num, Categorical = True):
    
    # Get the names of Independent and Dependent variables
    independent_variable_column = [df.columns[indep_column_num]]
    dep_variable = dependent_variable_name
    
    # Check for Target variable type to decide which CHAID TREE to implement
    if Categorical == True:
        
        # fit the Chaid tree model to supervised merged the categories in category predictor
        tree = Tree.from_pandas_df(df, dict(zip(independent_variable_column, [Predictor_type] *1)), 
                                   dep_variable, max_depth = 1)
        
    else:
        
        # Convert the target variable to numeric  
        df[dependent_variable_name] = pd.to_numeric(df[dependent_variable_name],errors='coerce')
        
        # fit the Chaid tree model to supervised merged the categories in category predictor
        tree = Tree.from_pandas_df(df, dict(zip(independent_variable_column, [Predictor_type] * 1)), 
                                   dep_variable, dep_variable_type='continuous', max_depth=1)
    
    # Print the fitted tree
    file.write('The CHAID TREE is presented below:\n\n')
    file.write(str(tree)+'\n')

    # Get the merged categoriess string from the tree
    Merged_group = tree.tree_store[0].split.groupings.split('],')
    # Get numbers of merged caegroeis
    length_Merged_group = np.arange(0,len(Merged_group))
    
    if len(Merged_group) >= 2: 
        
        # Etract the number from the string 
        New_Merged_Categories = {}
        for i in length_Merged_group:
            group = list(map(int, re.findall(r'\d+',Merged_group[i])))
            New_Merged_Categories[i] = group 
        file.write('The P-Values of this node is '+str(tree.tree_store[0].split.p)+'\n')
        file.write('The new categories are:\n')
        for k, v in New_Merged_Categories.items():
            file.write(str(k) + ' >>> '+ str(v) + '\n')
        
        # Convert the dict_format to match the previous dic
        # For example: new_merged: {0:[1,2,3,4,5],1:[6,7,8],2:[0,9]}
        #              map_dict: {0:2, 1:0, 2:0, 3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 9:2}               
        new_dict={}
        length_New_Merged = np.arange(0,len(New_Merged_Categories))
        for j in length_New_Merged:
            values = New_Merged_Categories.get(j) 
            for k in np.arange(0,len(values)):
                new_dict[values[k]]=j
    else:
        file.write('The P-Values of this node is '+str(tree.tree_store[0].split.p)+'\n')
        file.write('The P-values is too large.\n')
        file.write('There is no categories can be merged in this variables.\n\n')
        new_dict={}

    return new_dict

# Function to Rearrange categories and Supervised Merged for Categorical Predictors
# dataset = original dataset, column_type = dataset includes the columns type, dep_variable_name = target name.
def Reorder_Categories (file,dataset,column_type):
    dep_variable_name = get_target(dataset,column_type)
    
    # Get the target column index
    T_colnumber = dataset.columns.get_loc(dep_variable_name)
    
    # Get the type of Target column
    Flag_type = 'Flag_Continuous'
    Flag_type1 = Flag_type in column_type.iloc[:,1].values
    
    # Get the row and column number of Dataset
    n_columns=np.arange(0,len(dataset.columns),1)
    length_data = len(dataset)-1
    
    
    # Loop through all columns 
    for i in n_columns:
        
        Predictor_type = column_type.iloc[i,1]
        
        # Check the type of Categorical predictor
        if Predictor_type == 'Nominal':
            
            Pre_type = 'nominal'
            
            # Get the total counts of each category in each column
            Column_name = dataset.columns[i]
            Count_Each_Level = dataset.iloc[:length_data-1,i].value_counts()
            
            # Sort the categories  
            Count_Each_Level = sort_data(Count_Each_Level)
            file.write('Column name: '+Column_name.upper()+'\n')
            file.write(str(Count_Each_Level.to_frame())+'\n\n')


            # Assign each category a number, starting from 0 to N, by counts.
            n_distinct = np.arange(0,len(Count_Each_Level),1)
            dict_Level={}
            for j in n_distinct:
                Level_name = Count_Each_Level.index[j]
                dict_Level[Level_name]= j 


            file.write('Reorder Categories:\n')
            for k, v in dict_Level.items():
                file.write(str(k) + ' >>> '+ str(v) + '\n')

            # Substitute orignal Categories to number
            dataset[Column_name] = dataset[Column_name].map(dict_Level)


            # Supervised Merged
            file.write('\nSupervised Merged:\n')
            
            New_Categories ={}
            if T_colnumber != i:
                
                # Check if target is Categorical or Continuous
                if Flag_type1 == True:
                    New_Categories = Supervised_Merged(file,dataset, Pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i, Categorical = False)
                else:
                    New_Categories = Supervised_Merged(file,dataset, Pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i)
            else:
                file.write('\n')
                file.write('This is the Target column.\n')
                
            # Check if there the New_Categories is empty set
            if len(New_Categories) != 0:
                dataset[Column_name] = dataset[Column_name].map(New_Categories)
                
            file.write('------------------------------------------------------------------------------------------------------------------')
            file.write('\n')
            
         
        # If the Predictor type is Ordinal
        if Predictor_type == 'Ordinal':

            Column_name = dataset.columns[i]
            pre_type ='ordinal'

            New_Categories ={}

            # Check if target is Categorical or Continuous
            if Flag_type1 == True:
                New_Categories = Supervised_Merged(file,dataset, pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i, Categorical = False)
            else:
                New_Categories = Supervised_Merged(file,dataset,pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i)
            # If the column dtype is object(but the values are number), but during the supervised merged, the values were marked as int, then
            # transform the column dtype from object to int.

            types1 = [type(k) for k in New_Categories.keys()]


            if len(New_Categories) != 0:
                if types1[0] == int:
                    dataset[Column_name] = dataset[Column_name].astype(int)
                    dataset[Column_name] = dataset[Column_name].map(New_Categories)
                else:
                    dataset[Column_name] = dataset[Column_name].map(New_Categories)
    return dataset

def p_value_continuous(column1,column2):
    column2=pd.to_numeric(column2,errors='ignore')
    F = np.var(column1) / np.var(column2)
    #degree of freedom
    df1 = len(column1) - 1
    df2 = len(column2) - 1
    p_value = round(stats.f.sf(F, df1, df2),5)
    return p_value

def p_value_target_predictor(target,column,df_type):
    target_name = target.name
    if column_type(target_name,df_type)=="Flag_Continuous":
        return (p_value_continuous(target,column))
    else:
        contingency = pd.crosstab(target, column)
        c, p, dof, expected=stats.chi2_contingency(contingency)
        return p

# Function to get best depth which help to train optimal model
# Input: data of each column, data of decision varaiable
# Output: best depth
def get_best_depth(file,d_column, d_flag):
    score_mean = [] # here I will store the roc auc
    depth_list = [1,2,3,4,5,6,7,8,9,10]    
    for depth in depth_list:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        # calculate roc_auc value
        score = cross_val_score(tree_model, d_column, d_flag, cv=3, scoring='roc_auc')    
        score_mean.append(np.mean(score))    
    # create a dataframe to store depth and roc_auc value
    table = pd.concat([pd.Series(depth_list), pd.Series(score_mean)], axis=1)
    table.columns = ['depth', 'roc_auc_mean']    
    # get best depth
    table_sort = table.sort_values(by='roc_auc_mean', ascending=False) 
    best_depth = table_sort.iloc[0,0] # get depth which lead ot the largest roc_auc
    file.write(str(table_sort)+'\n')
    file.write('Best Depth:'+str(best_depth)+'\n')
    return (best_depth) 


# Function to do supervised binning, based single variable decision tree model
# Input: all data processed in the previous step，the data including variable and column type
# Output: new data file
def supervised_binning(file,df,df_type):    
    # get all continuous variable
    new_df = get_continuous_variables(df,df_type)
    # get target
    d_flag = df[[get_target(df,df_type)]] # get data of target
    #d_flag.loc[d_flag['Claim_Amount'] != 0] = 1
    column_list = new_df.columns.values.tolist()
    num_row = len(new_df)

    i = 0
    for column in column_list:
        
        # get data of a certain column
        d_column = new_df[[column]]
        
        num_unique = len(new_df[column].unique())
        
        # select best parameter (max_depth, max_leaf_node) 
        if num_row <= 10000:
            num_bins = None
            depth = get_best_depth(file,d_column,d_flag)
            file.write("do not set 'max_leaf_node'\n")
        elif (num_row >= 10000 and num_unique <=64):
            num_bins = None
            depth = get_best_depth(file,d_column,d_flag)
            file.write("do not set 'max_leaf_node'\n")
        else:
            depth = None
            num_bins = int(np.sqrt(num_unique))
            file.write("do not set 'max_depth'\n")
         
        # train optimal single variable to do supervised binning
        optimal_model = DecisionTreeClassifier(max_depth=depth,max_leaf_nodes=num_bins)
        optimal_model.fit(d_column, d_flag)
        y_pred = optimal_model.predict_proba(d_column)[:,1]
        score = roc_auc_score(d_flag,y_pred)
        df[column]=y_pred
        file.write('Column name:'+str(column)+'\n')
        file.write('The number of original unique value (bins):'+str(num_unique)+'\n')
        file.write('The number of unique value (bins):'+str(len(df[column].unique()))+'\n')
        file.write('The value of each bins:'+str(df[column].unique())+'\n')
        file.write('Roc_Auc value:'+str(score)+'\n\n')
        i=i+1
    return (df) 

#function to get all continuous variables. Preparing for PCA
#input: a record dataset and a column type dataset after predictors handling
#output: a dataset contains only continuous variables
def get_continuous_variables(new_df,new_df_type):
    continuous_predictor_name = []
    for c in new_df:
        if column_type(c,new_df_type) == "Continuous":
            continuous_predictor_name.append(c)
    return (new_df[continuous_predictor_name])

#function to delete non highly correlated features with target
def continuous_selection(new_df,new_df_type):
    target_name = get_target(new_df,new_df_type)
    target = new_df[target_name]
    df_continuous_predictors = get_continuous_variables(new_df,new_df_type)
    for c in df_continuous_predictors:
        if p_value_target_predictor(target,df_continuous_predictors[c],new_df_type) > 0.05:
            df_continuous_predictors = df_continuous_predictors.drop(columns = c)
            new_df = new_df.drop(columns = c)
            new_df_type = new_df_type.drop(index=int(new_df_type[new_df_type.Variable == c].index[0]))
    return (df_continuous_predictors,new_df,new_df_type)

#function to get correlation between a feature and a group
def get_corr_group(variable_index,group_list,new_continuous_predictors):
    corr_list = []
    for i in group_list:
        matrix = np.corrcoef(new_continuous_predictors.iloc[:,variable_index],new_continuous_predictors.iloc[:,i])
        corr_list.append(abs(matrix[0,1]))
    return min(corr_list)

#function to get grouped feature index. 
#input: a datset of all continuous variables. 
#output: a list of names of grouped features.
def get_grouped_features(new_continuous_predictors):
    #correlation matrix
    corre = new_continuous_predictors.corr()
    #triangular matrix
    tri_corre = np.triu(corre)
    #changed to absolute values
    tri_corre = np.absolute(tri_corre)
    #changed correlation to 0 for correlation of feature itself
    for i in range(len(tri_corre)):
        tri_corre[i,i] = 0
    groups = []
    alpha = 0.9
    while alpha > 0.1:
        #get first pair in a group
        if (np.amax(tri_corre) > alpha):
            group = list(np.unravel_index(np.argmax(tri_corre), (len(tri_corre),len(tri_corre))))

            #max features in a group is 5
            while len(group) <= 5:
                #get the next correlated feature to group
                group_var_corr = {}
                for i in range(new_continuous_predictors.shape[1]):
                    if i in group:
                        continue
                    else:
                        group_var_corr[i] = get_corr_group(i,group,new_continuous_predictors)
                best_var_index = max(group_var_corr, key=group_var_corr.get)
                if group_var_corr[best_var_index] > alpha:
                    #add feature i to the group
                    group.append(i)

                else:
                    #remove correlations of grouped features
                    group_name = []
                    for i in group:
                        tri_corre[i] = 0
                        tri_corre[:, i] = 0
                        group_name.append(new_continuous_predictors.iloc[:,i].name)                
                    groups.append(group_name)
                    break

        else:
            alpha -= 0.1
    return(groups)

#function to get continuous features after PCA
#input: a dataset contains only continuous features with high correlation with target and the new_df from previous steps
#output: a dataset contains only continuous features after PCA 
def get_continuous_after_pca(new_continuous_predictors,new_df,new_df_type):
    pca = PCA(n_components=1)
    groups = get_grouped_features(new_continuous_predictors)
    for group in groups:
        #get new feature name
        for i in range(len(group)):
            if i == 0:
                new_pca = str(group[i])
            else:
                new_pca = new_pca + '_' + str(group[i])
        df_pca = new_continuous_predictors[group]
        pca.fit(df_pca)
        X_pca=pca.transform(df_pca) 
        X_pca = pd.DataFrame(X_pca)
        new_df = new_df.drop(columns=group)
        for c in group:
            new_df_type = new_df_type.drop(index=int(new_df_type[new_df_type.Variable == c].index[0]))
        new_df[new_pca] = X_pca
        new_df_type = new_df_type.append(pd.DataFrame([[new_pca,'Continuous']], columns=['Variable','Type']))
    return(new_df,new_df_type)

def main():
    #load data
    data_file_name = input('Data file name: ')
    data_type_file_name = input('Column type file name: ')
    df = pd.read_csv('../../data/raw/'+data_file_name+'.csv')
    df_type = pd.read_csv('../../data/raw/'+data_type_file_name+'.csv')

    target_name = get_target(df,df_type)
    target_type = column_type(target_name,df_type)

    # Basic screening and print report
    file = open('../../reports/build_features/'+data_file_name+'_raw_univariate_statistic_report.txt','w') 
    d = Stats_Collection(file,df,df_type)
    file.close()
    # After deleting useless variables, new dataset and new column type dataset are named as new_df and new_df_type
    #new_df & new_df_type do not contain target
    new_df_type = d[1]
    new_df = d[0]

    #Outlier handling
    for c in new_df:
        if column_type(c,new_df_type) == 'Continuous':
            outlier_trim(new_df,new_df[c])

    #Missing value handling
    new_df = new_df.replace('?',np.NaN)
    fill_missing_value(new_df,new_df_type)

    #Continuous variable transformation, choose between zscore and minmax
    for c in new_df:
        new_df[c] = zscore(new_df, c,new_df_type)
        # new_df[c] = minmax(new_df, c,new_df_type)

    #Box-Cox transformation of continuous target
    new_df[target_name] = boxcox(new_df, target_name, new_df_type)


    #Categorical variable handling: Reorder and Supervised Merged
    file = open('../../reports/build_features/'+data_file_name+'_supervised_merge_report.txt','w') 
    new_df = Reorder_Categories(file,new_df,new_df_type)
    file.close()

    #Continuous variable handling when target is continuous: Abandon not highly correlated features with target
    if target_type == 'Flag_Continuous':
        selection = continuous_selection(new_df,new_df_type)
        new_continuous_predictors = selection[0]
        new_df = selection[1]
        new_df_type = selection[2]

        #Continuous variable construction:
        #If there are more than 2 continuous variables left in data set, perform PCA
        if new_continuous_predictors.shape[1] != 0:
            pca = get_continuous_after_pca(new_continuous_predictors, new_df, new_df_type)
            new_df = pca[0]
            new_df_type = pca[1]

    #Continuous variable handling when target is categorical
    else:
        file = open('../../reports/build_features/'+data_file_name+'_supervised_binning_report.txt','w')
        new_df = supervised_binning(file,new_df,new_df_type)
        file.close()

    new_df.to_csv('../../data/processed/processed_'+data_file_name+'.csv', index=False)
    new_df_type.to_csv('../../data/processed/processed_'+data_type_file_name+'.csv', index=False)

    #Univariate stats report for processed data
    file = open('../../reports/build_features/'+data_file_name+'_processed_univariate_statistic_report.txt','w')
    d = Stats_Collection(file,new_df,new_df_type)
    file.close()


main()

# elapsed = (time.clock() - start)
# print("Time used:",elapsed)
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
