import pandas as pd
import numpy as np
from scipy import stats
import sklearn
from sklearn import preprocessing
from CHAID import Tree
import re
from sklearn.decomposition import PCA



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
    return min(column)

#function to get maximum value in a column
def get_max(column): #input the whole column
    return max(column)

#function to get mean in a column
def get_mean(column): #input the whole column
    return column.mean()

#function to get std in a column
def get_std(column): #input the whole column
    return column.std()

#function to get skewness in a column
def get_skew(column): #input the whole column
    return column.skew()

#function to get number of distict values in a column
def get_distinct_num(column): #input the whole column
    return len(column.unique().tolist())

#function to get count of each distict value in a column
def get_distinct_count(column): #input the whole column
    if get_distinct_num(column) > 5:
        print('Number of distict values is larger than 5. We stop updating the number of distinct values')
    else:
        return column.value_counts()

#function to get median in a column
import statistics
def get_median(column):
    if get_distinct_num(column) > 5:
        print('Number of distict values is larger than 5. We do not calculate median')
    else:
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
def Stats_Collection(df,df_type):
    for c in df:
        #exclude Target 
        if (column_type(c,df_type) != 'Flag_Continuous' and column_type(c,df_type) != 'Flag_Categorical'):
            print('Variable name: ',c)

            #Basic variable screening
            if get_na_num(df[c])/len(df[c]) > 0.5:
                print('More 50% missing values, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Delete'):
                print('Column type is Delete, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Continuous') and (get_min(df[c]) == get_max(df[c])):
                print('All same value, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Ordinal' or column_type(c,df_type) == 'Nominal') and (get_mode(df[c])[1]/get_valid_num(df[c]) > 0.95):
                print('Mode contains more than 95% cases, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue
            if (column_type(c,df_type) == 'Nominal') and (get_distinct_num(df[c]) > 100):
                print('More than 100 categories, drop this column\n')
                df = df.drop(columns=c)
                df_type = df_type.drop(index=int(df_type[df_type.Variable == c].index[0]))
                continue

            #Basic statistic report
            print('Variable type: ', column_type(c,df_type))
            print ('Number of missing values: ',get_na_num(df[c]))
            print ('Number of valid values: ',get_valid_num(df[c]))
            if column_type(c,df_type) == 'Continuous' or column_type(c,df_type) == 'Ordinal':
                print('Minimum value: ', get_min(df[c]))
                print('Maximum value: ', get_max(df[c]))
            if column_type(c,df_type) == 'Continuous':
                print('Mean: ',get_mean(df[c]))
                print('Standard Deviation: ',get_std(df[c]))
                print('Skewness: ',get_skew(df[c]))
                print('Number of distinct values: ',get_distinct_num(df[c]))
                print('Number of cases for each distinct value: ')
                print(get_distinct_count(df[c]))
            else:
                print('Number of categories: ', get_distinct_num(df[c]))
                print('The counts of each category: ')
                print(get_distinct_count(df[c]))
                print('Mode: ', get_mode(df[c])[0],'Count: ',get_mode(df[c])[1])                
        print()
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
    result = (column - x_robust < -3 * sd_robust) | (column + x_robust > 3 * sd_robust)
    lower_cutoff_value = x_robust - 3 * sd_robust
    upper_cutoff_value = x_robust + 3 * sd_robust
    return(result, lower_cutoff_value, upper_cutoff_value)

# function to trim outliers to cutoff values
def outlier_trim(df,column):
    (flag, lower, upper) = outlier_identification(column)
    df.loc[flag, column.name] = pd.Series(map(lambda x : (lower if x <= lower else upper), column[flag]))

# function to set outliers to missing values
def outlier_toNone(df,column):
    (flag, lower, upper) = outlier_identification(column)
    df.loc[flag, column.name] = None

# function to fill missing value and update statistic
def fill_missing_value(mydata, column_type):
    # filling missing value and updata statistic
    column_list = mydata.columns.values.tolist()
    typelist = list(column_type.iloc[:,1])
    i = 0
    for typ in typelist:
        column_data = mydata[column_list[i]].dropna()
        if (typ == 'Continuous'):
            mean_value = column_data.mean() # calculate mean
            mydata[column_list[i]] = mydata[column_list[i]].fillna(mean_value) # fill missing value with mean
            cont_sd = mydata[column_list[i]].std() # calculate standard deviation
            cont_skew = mydata[column_list[i]].skew() # calculate skewness
            print('')
            print('Column:', column_list[i])
            print('Column type: continuous')
            print('Mean:', mean_value)
            print('Standard deviation:', cont_sd)
            print('Skewness:', cont_skew)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))
        elif (typ == 'Ordinal' and mydata[column_list[i]].dtype != 'object'):
            num_median = column_data.median() # calculate median 
            mydata[column_list[i]] = mydata[column_list[i]].fillna(num_median) # fill missing value with median
            count_median_num = mydata[column_list[i]][mydata[column_list[i]] == num_median].count() # count the the number of cases in the median category
            print('')
            print('Column:', column_list[i])
            print('Column type: num_ordinal')
            print('Median:', num_median)
            print('The number of cases in the median category:', count_median_num)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))
        elif (typ == 'Ordinal' and mydata[column_list[i]].dtype == 'object'):
            mode_value = column_data.mode()[0] # calculate mode
            mydata[column_list[i]] = mydata[column_list[i]].fillna(mode_value) # fill missing valye with mode
            count_mode = mydata[column_list[i]][mydata[column_list[i]] == mode_value].count() # count the the number of cases in the modal category
            print('')
            print('Column:', column_list[i])
            print('Column type: cat_ordinal')
            print('Mode:', mode_value)
            print('The number of cases in the modal category:', count_mode)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))
        else:
            mode_value = column_data.mode()[0] # calculate mode
            mydata[column_list[i]] = mydata[column_list[i]].fillna(mode_value) # fill missing valye with mode
            count_mode = mydata[column_list[i]][mydata[column_list[i]] == mode_value].count() # count the the number of cases in the modal category
            print('')
            print('Column:', column_list[i])
            print('Column type: nominal')
            print('Mode:', mode_value)
            print('The number of cases in the modal category:', count_mode)
            print('The number of missing values:', get_na_num(mydata[column_list[i]]))
            print('The number of valid values:', get_valid_num(mydata[column_list[i]]))
        i = i + 1
    # add column type at the last row
    print('add column type at the last row:')
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

# Function to sort the Series by value and then by index(Lexical order)
def sort_data(Series):
    return Series.iloc[np.lexsort([Series.index, Series.values])]


# Function to supervised merged categories in categorical variables
# df = dataset, Predictor_type = Nominal or Ordinal, dependent_variable_name = target name, indep_column_num = column index
def Supervised_Merged (df, Predictor_type, dependent_variable_name, indep_column_num, Categorical = True):
    
    
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
                                   dep_variable, dep_variable_type='continuous',max_depth = 1)
    
    # Print the fitted tree
    print('The CHAID TREE is presented below:')
    print('')
    tree.print_tree()

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
        print('The P-Values of this node is',tree.tree_store[0].split.p)
        print('The new categories are:' )
        print(New_Merged_Categories)
        print('')
        
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
        print('The P-Values of this node is',tree.tree_store[0].split.p)
        print('The P-values is too large.')
        print('There is no categories can be merged in this variables.')
        print('')
        new_dict={}
    return new_dict

# Function to Rearrange categories and Supervised Merged for Categorical Predictors
# dataset = original dataset, column_type = dataset includes the columns type, dep_variable_name = target name.
def Reorder_Categories (dataset,column_type):
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
            print('Column name:',Column_name.upper())
            print(Count_Each_Level.to_frame())


            # Assign each category a number, starting from 0 to N, by counts.
            n_distinct = np.arange(0,len(Count_Each_Level),1)
            dict_Level={}
            for j in n_distinct:
                Level_name = Count_Each_Level.index[j]
                dict_Level[Level_name]= j 


            print('Reorder Categories :')
            print(dict_Level)
            print('')

            # Substitute orignal Categories to number
            dataset[Column_name] = dataset[Column_name].map(dict_Level)


            # Supervised Merged
            print('Supervised Merged:')
            
            New_Categories ={}
            if T_colnumber != i:
                
                # Check if target is Categorical or Continuous
                if Flag_type1 == True:
                    New_Categories = Supervised_Merged(dataset, Pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i, Categorical = False)
                else:
                    New_Categories = Supervised_Merged(dataset, Pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i)
            else:
                print('')
                print('This is the Target column.')
                
            # Check if there the New_Categories is empty set
            if len(New_Categories) != 0:
                dataset[Column_name] = dataset[Column_name].map(New_Categories)
                
            print('------------------------------------------------------------------------------------------------------------------')
            print('')
            
         
        # If the Predictor type is Ordinal
        if Predictor_type == 'Ordinal':
            
            Column_name = dataset.columns[i]
            Pre_type ='ordinal'
            
            New_Categories ={}
            
            # Check if target is Categorical or Continuous
            if Flag_type1 == True:
                New_Categories = Supervised_Merged(dataset, Pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i, Categorical = False)
            else:
                New_Categories = Supervised_Merged(dataset,Pre_type, dependent_variable_name = dep_variable_name, indep_column_num = i)
                
            if len(New_Categories) != 0:
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
    return df_continuous_predictors

#function to get correlation between a feature and a group
def get_corr_group(variable_index,group_list,new_continuous_predictors):
    corr_list = []
    for i in group_list:
        matrix = np.corrcoef(new_continuous_predictors.iloc[:,variable_index],new_continuous_predictors.iloc[:,i])
        corr_list.append(matrix[0,1])
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
def get_continuous_after_pca(new_continuous_predictors,new_df):
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
        new_df[new_pca] = X_pca
    return(new_df)

def main():
    #load data
    df = pd.read_csv('data/raw/Kaggle/train_sample.csv')
    df=df.drop(columns = ['Unnamed: 0']) #only for this sample dataset
    df_type = pd.read_csv('data/raw/Kaggle/column_type.csv')

    # Basic screening and print report
    d = Stats_Collection(df,df_type)
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

    #Categorical variable handling: Reorder and Supervised Merged
    new_df = Reorder_Categories(new_df,new_df_type)

    #Continuous variable handling Supervised Binning


    #Continuous variable handling when target is continuous: Abandon not highly correlated features with target
    new_continuous_predictors = continuous_selection(new_df,new_df_type)

    #Continuous variable construction:
    new_df = get_continuous_after_pca(new_continuous_predictors,new_df)

    print(new_df.head())

main()



