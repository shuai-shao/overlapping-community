# read in raw data of stock prices of S&P 500 companies and build correlation
# networks from returns


import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import glob 

return_dict = {}    # return_dict to store name and stock_return for each company
company_index = {}  # a dict of company names and their index 
path = r'/Users/seanshao/Documents/research/syn/ss/stockFilesLong/'    #folder name
len_path = len(path)    #length of path 
allfiles_rrr = glob.glob(path+'*')      # allfiles contains all files in the fold
allfiles_rr = []
for files in allfiles_rrr:
    allfiles_rr.append(files[len_path:])
d_list = pd.read_csv('/Users/seanshao/Documents/research/syn/ss/del_list.csv')    # delelte companies don't exist in sp500 anymore
del_list = list(d_list['symbol'])
allfiles_r=[x for x in allfiles_rr if x not in del_list] #del_list is the list of 27 companies that do not exist in stock market any more
allfiles = []
for files in allfiles_r:
    df = pd.read_csv(path+files)
    if len(df) == 2306:    #  only keep stocks with full length of time
        allfiles.append(files)

 
indexing = 0 
for files in allfiles:
    df = pd.read_csv(path + files)  
    company_index[indexing] = files  # index for each company 
    indexing = indexing + 1 
    stock_price = np.array(df['Adj Close'])    # np array for stock price
    stock_return = np.zeros(len(stock_price)-1)
    for i in range(len(stock_price)-1):
        stock_return[i] = (stock_price[i+1]-stock_price[i])/stock_price[i]    #calculate arithmetic return 
    return_dict[files] = stock_return    # store name of company and array of stock prices
    
num_company = len(company_index)
cor_matrix = np.zeros((num_company,num_company))    # correlation matrix

# now calculate the correlation matrix 
for i in range(num_company):
    company_name_i = company_index[i]    #name of i th company
    return_i = return_dict[company_name_i]    # the return array of i the company
    for j in range(num_company):
        company_name_j = company_index[j]    # name of j the company
        return_j = return_dict[company_name_j]
        cor_matrix[i][j] = pearsonr(return_i,return_j)[0]    #smallest indexes indicate most recent returns, so this is valid


# write cor_matrix to file
df = pd.DataFrame(cor_matrix, columns = allfiles, index = allfiles)
df.to_csv('/Users/seanshao/Documents/research/syn/ss/cor_coef.csv')

