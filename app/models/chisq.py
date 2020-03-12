


def __simple_chi (df, column):
    from scipy import stats
    temp = df[column].value_counts()
    chi = stats.chisquare(temp)
    return 100*(1-chi[1])


def group_chi (df, group, column):
    return df.groupby(group).apply(__simple_chi, column).reset_index().rename(columns={0 : column+'Chi'})


# import pandas as pd

# data = pd.read_csv('kdemo.tab', sep = '\t')

# print(group_chi(data, 'DLR_KEY', 'ProductCodeDescr'))
