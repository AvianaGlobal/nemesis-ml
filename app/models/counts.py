
# count number of data points in a series
def group_size(series):
    return len(series)


# count number of data points in each group in a series
def count_by_group(data, group, score):
    return data.groupby(group).count()[[score]]