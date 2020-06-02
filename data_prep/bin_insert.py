
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os

def bin_insert(df,target_col,groupby_col):

    if is_numeric_dtype(df[target_col]) == True:
        ###overview###
        n = df[target_col].count() 
        print('The variable ' + str(target_col)+ ' have ' + str(n) + ' records')
        ###break down into bins###
    
        redo ='Y'
        data = pd.DataFrame(columns = df.columns) 
        while redo.upper() == 'Y':
            nbin = input('How many bins would like to have? Insert a integer')
            nbin = int(nbin)

            if nbin > 0:
                
                binrg = round(n/nbin)
                i = 1
                while i <= nbin-1:
                    df1 = df[int((i-1)*binrg):int(i*binrg)] 
                    df1 = df1.assign(bin = int(i))
                    data = data.append(df1)
                    i = int(i + 1)
                df2 = df[int((nbin-1)*binrg):n]
                df2 = df2.assign(bin = nbin)
                data = data.append(df2)
                print(data)
                data.to_csv('Backup.csv')

                print('Start grouping....')
                ###grouping###
                g = 1
                while g <= nbin:
                    count = data[data['bin'] == g].groupby(groupby_col).count()[['bin']]
                    count = count.rename(columns = {'bin':'bin_is_' + str(g) +'within_group'})
                    data = pd.merge(data,count,on = groupby_col)
                    g = g + 1

                print("Here's the new data! \n")
                print(data)

                os.remove("Backup.csv")
                data.to_csv('Newdata_BinCreated.csv')
                return data

            else:
                print('Please enter a vaild positive integer')
                redo ='Y' 
            
        
    else:
        print(str(target_col) + ' is not numeric.')
        
        




