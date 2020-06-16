# this code is Anderson-Darling Normality test to see if the distribution is normal.

# The code for importing the sample_data is not here. If it is added in the future, this comment will be deleted

# Anderson-Darling normality Test
from scipy.stats import anderson

if __name__ == '__main__':
    result = anderson(data)
    print(result)
