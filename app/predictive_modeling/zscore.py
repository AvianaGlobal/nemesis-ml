from scipy import stats


# function to do z-score transformation of a column
def zscore(df, column, df_type):
    if column_type(column, df_type) == 'Continuous':
        return (stats.zscore(df[column]))
    else:
        return (df[column])
