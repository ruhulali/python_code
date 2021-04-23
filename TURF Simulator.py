# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:52:20 2020
 
@author: ruhul
"""
#---------------------------------------------------------------------
### Import Packages
#import ray
#ray.init(num_cpus=4)
#import modin.pandas as pd
 
import pandas as pd
import numpy as np
import os
from itertools import combinations
import time
os.chdir('D:/KNOWLEDGE KORNER/OFFICE/Simulator/Raw_Data')
 
#---------------------------------------------------------------------
### Load Dataset
df = pd.read_excel("raw_data_copy.xlsx")
df.rename(columns={ df.columns[0]: "ID" }, inplace = True)
df.set_index('ID', inplace=True)
df = df.astype(np.float32)
df.info()
 
df_filter = pd.read_excel("filters.xlsx")
df_filter.rename(columns={ df_filter.columns[0]: "ID" }, inplace = True)
df_filter.set_index('ID', inplace=True)
df_filter.info()
 
#database = df_filter[df_filter['Income'] == 1]
#database = database.merge(df, how='inner', left_index=True, right_index=True).iloc[:,df_filter.shape[1]+1:]
 
### Function to merge raw data with filter data
def merge_data(raw_data, filter_data):	
	#database = df_filter[df_filter['Income'] == 1]
	database = filter_data.merge(raw_data, how='inner', left_index=True, right_index=True)#.iloc[:,df_filter.shape[1]+1:]
	columns = list(filter_data.columns)
	return database, columns
 
#def filter_function(raw_data, filter_data, *args):
#	data, columns = merge_data(raw_data, filter_data)
#	args_passed = len(args)
#	for i in range(0,args_passed):
#    	if  args[i] != 'All':
#        	data = data[data.iloc[:,i] == args[i]]
#	data = data.iloc[:,len(columns)+1:]
#	return data
 
### Function for filtering
def filter_function(raw_data, filter_data, filters):
	data, columns = merge_data(raw_data, filter_data)
	filters_passed = len(filters)
	for i in range(0,filters_passed):
    	if  filters[i] != 'All':
        	data = data[data.iloc[:,i] == filters[i]]
	data = data.iloc[:,len(columns)+1:]
	return data
 
#final= filter_function(df, df_filter, ['All',3])
        	
### Function for Combination
def result(combination, df, df_filter, top, no_attr, attr_filter):
	"""
	combination: Number of combinations
	df: Raw Scores
	df_filters: Filter data
	top: Top number of combinations
	no_attr: Attribute shown per screen
	attr_filter: Filter values
	"""
	database = filter_function(df, df_filter, attr_filter)
	
	num_attr = database.shape[1]
	comb = list(combinations(database.columns.values.tolist(), combination))
	#comb = np.array(combinations(database.columns.values.tolist(), combination))
	database = database.copy()
	for items in comb:
    	database[' & '.join(items)] = np.exp(database.loc[:, items].values).sum(1)/((np.exp(database.loc[:, items].values).sum(1)+(no_attr-1)))
    	
	database_ = database.copy()
	if database_.shape[1]>num_attr:
    	Result = database_.iloc[:,num_attr:].mean(0)
	else:
    	Result = database_.mean(0)
 
	Result_Reach = pd.DataFrame(Result)
    Result_Reach.reset_index(inplace = True)
	Result_Reach.columns = ['Feature Combinations','Percentage Reach']
	Result_Reach['Percentage Reach'] = Result_Reach['Percentage Reach']*100
	Result_Reach['Percentage Reach'] = Result_Reach['Percentage Reach'].round(decimals=2)
    Result_Reach['Combination_No'] = combination
	#Result_Reach["Filter"] = list(attr_filter)
	Result_Reach['Base Size'] = len(database)
	Result_Reach['Base Size'] = Result_Reach['Base Size'].astype(np.uint8)
    Result_Reach['Combination_No'] = Result_Reach['Combination_No'].astype(np.uint8)
	Result_Reach_top = Result_Reach.nlargest(top,['Percentage Reach'])
	return Result_Reach_top
 
# Time taken to create combinations
start = time.time()
print("Creating combination....")
#--------
New_results = pd.DataFrame()
for i in [["All","All"],[1,"All"],[1,1],[1,2],[1,3],[1,4],[1,5],[2,"All"],[2,1],[2,2],[2,3],[2,4],[2,5]]:
	temp = result(6, df, df_filter, 100, 4, i)
	#New_results = pd.concat([New_results,temp], axis=0)
	New_results = New_results.append(temp, ignore_index=True)
#--------
end = time.time()
print("Combination Produced. \nTime Taken (in seconds)", end - start)
 
# for i in [['All','All'],['All',1],['All',2],['All',3],['All',4],['All',5],[1,'All'],[1,1],[1,2],[1,3],[1,4],[1,5],[2,"All"],[2,1],[2,2],[2,3],[2,4],[2,5]]:
 
# Export Excel
New_results.to_excel('Comb_Result_6_2Filters.xlsx', index=False)
 
### Export Excel
New_results.to_excel('Comb_Results_4_100_Non-Hispanic_All.xlsx', index=False)
New_results.head()
New_results.info()
 
#-----------------------------
# 4
35 / 24 / 20
 
# 6
2051 / 1178 / 1151
 
# 8
28167 / 28120
 
#---------------------------------
 
# total combinations to run including filter
#all all
#all 1
#all 2
#all 3
#all 4
#all 5
#1       	all
#1       	1
#1       	2
#1       	3
#1       	4
#1       	5
#2       	all
#2       	1
#2       	2
#2       	3
#2       	4
#2       	5
 
