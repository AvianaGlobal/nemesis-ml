# count number of sample_data points in a series
def group_size(series):
    return len(series)


# count number of sample_data points in each group in a series, it returns a sample_data frame with the group and its size
def count_by_group(df, group, entity):
    return df.groupby(df[group]).count()[entity].reset_index().rename(columns={entity: 'Size'})
