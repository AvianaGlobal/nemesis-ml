# allow users to enter their filter
# data should be a pandas dataframe and criterion should follow pandas syntax

def filter(data, criterion):
    data = data[criterion]
    return data