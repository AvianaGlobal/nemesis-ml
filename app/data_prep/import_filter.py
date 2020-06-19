# allow users to enter their filter
# data should be a pandas dataframe and criterion should follow pandas syntax

def data_filter(data, criterion):
    data = data.query(criterion).reset_index(drop = True)
    print('Filter: ' + str(criterion) + ' applied')
    print(' ')
    print(data)
    return data