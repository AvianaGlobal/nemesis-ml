

def simple_chi (df, column):
    from scipy import stats
    temp = df[column].value_counts()
    chi = stats.chisquare(temp)
    return chi


def pearson_redisual (df, col1, col2):
    r_sq = map(lambda i, j: ((i-j)**2)/(j+0.00000000000001), df[col1], df[col2])
    return sum(list(r_sq))

# import pandas as pd

# data = pd.read_csv('card.csv')

# print(pearson_redisual(data, 'count_Cardnum_7d', 'Actual/mean_Merchnum_14d'))