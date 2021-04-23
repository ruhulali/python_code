# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:52:13 2021

@author: Ruhul.Akhtar
"""

#TURF Simulator - Milan

import pandas as pd
import os
os.chdir('D:/KNOWLEDGE KORNER/OFFICE/Simulator/Raw_Data')
 
 
### Load Dataset
df_Raw = pd.read_csv("data/Raw_file.csv")
df_Raw.columns
df_Raw.rename(columns={df_Raw.columns[0]: "ID" }, inplace = True)
df_Raw.set_index('ID', inplace=True)
# df.head()
# df.info()
 
df_filter = pd.read_csv("data/Filter_file.csv")
df_filter.rename(columns={ df_filter.columns[0]: "ID" }, inplace = True)
df_filter.set_index('ID', inplace=True)
# df_filter.columns
 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
import pandas as pd
import numpy as np
from itertools import combinations
 
def Recode_Scale(series,num):
	if series >= num:
    	return 1
	else:
    	return 0
 
def Rating(df,num):
	df_working = df.copy()
	num_attr = df_working.shape[1]
	for i in df_working:
    	df_working[f'{i} '] = df_working[f'{i}'].apply(Recode_Scale,num=num)
	return df_working.iloc[:,num_attr:]
 
### Function for Filter
def merge_data(raw_data, filter_data):
	#raw_data = Rating(raw_data)	
	#database = df_filter[df_filter['Income'] == 1]
	database = filter_data.merge(raw_data, how='inner', left_index=True, right_index=True)#.iloc[:,df_filter.shape[1]+1:]
	columns = list(filter_data.columns)
	return database, columns
 
def filter_function(raw_data, filter_data, main_filter,second_filter):
	data, columns = merge_data(raw_data, filter_data)
	if  main_filter != 'All Respondents' and len(second_filter) != 0:
    	data = data[data[main_filter].isin(second_filter)]
	data = data.iloc[:,len(columns):]
	return data
 
############## Combination Function
def Binnary_result(combination, df, df_filter, top, main_filter,second_filter):
	"""
	combination: Number of combinations
	df: Raw Scores
	df_filters: Filter data
	top: Top number of combinations
	no_attr: Attribute shown per screen
	attr_filter: Filter values
	"""
	database = filter_function(df, df_filter, main_filter,second_filter)	
	num_attr = database.shape[1]
	comb =  list(combinations(database.columns.values.tolist(), combination))
    	
	database = database.copy()  
	for items in comb:
    	database[' & '.join(items)] = database.loc[:, items].values.sum(1)
 
	database_ = database.replace(0, np.nan)
	
	if combination > 1 :
    	Result = database_.iloc[:,num_attr:].count()
	else:
    	Result = database_.count()
	Result_Reach = pd.DataFrame(Result)	
    Result_Reach.reset_index(inplace = True)
	Result_Reach.columns = ['Feature Combinations','Frequency']
    Result_Reach.sort_values(['Frequency'], ascending = False, inplace = True)
	Result_Reach['Percentage Reach'] = (Result_Reach['Frequency']/len(database))*100
	Result_Reach['Percentage Reach'] = Result_Reach['Percentage Reach'].round(decimals=2)
	# Result_Reach['Combination_No'] = combination
	# Result_Reach['Combination_No'] = Result_Reach['Combination_No'].astype(np.uint8)
	Result_Reach_top = Result_Reach.nlargest(top,['Percentage Reach'])
	return Result_Reach_top
 
 
def make_id_sets(df, df_filter, main_filter,second_filter):
	"""
	Makes a list of sets of user IDs corresponding to individuals who selected that feature or reason. Need index to be user id,
	doesn't matter what the columns are named. List of sets will be the same order as that contained in Dataset columns.
	"""	
	dataframe = filter_function(df, df_filter, main_filter,second_filter)
	sets = []
	for (column_name,data) in dataframe.iteritems():
    	to_set = set(dataframe.index[data == 1])
    	sets.append(to_set)
	return sets
 
def reach_percentage_and_order(sets,df, df_filter, main_filter,second_filter):
	"""Initaties two lists, unduplicated reach and feature order  using starting index value and Dataset"""
	dataframe = filter_function(df, df_filter, main_filter,second_filter)
	score = dataframe.sum()
	score = score.to_frame(name=None)
	Top_Att = score[score[0]==dataframe.sum().max()].index.values
	Attributes = dataframe.columns.values.tolist()
	for n, i in enumerate(Attributes):
    	for j in Top_Att:
        	if j in i:
            	position = n
	
	# if position == 0:
	# 	position = 1	
	
	return [((len(sets[position]))/(len(dataframe)))], [dataframe.columns[position]]
 
def calculate_order_percentages(sets,df, df_filter, main_filter,second_filter):
	"""
	First calls functions to set the starting point for the TURF. Then will loop through the range specified (this is the
	number of features to go through before you reach the limit you're looking for). Each outer loop has an inner loop that
	checks the difference between the full set of features currently held and the set in that iteration. Set with the most
	difference has its index value added and is joined with the full set. Reach percentage is also calculated. Returns
	order and percentages as a list.
	
	***Starting feature index will be 1 less than feature number unless you're starting at 0. ie Feature 8 is at 7 index.
	""" 
	percentages, order = reach_percentage_and_order(sets,df, df_filter, main_filter,second_filter)
	
	dataframe = filter_function(df, df_filter, main_filter,second_filter)  
	score = dataframe.sum()
	score = score.to_frame(name=None)
	
	Top_Att = score[score[0]==dataframe.sum().max()].index.values
	Attributes = dataframe.columns.values.tolist()
	for n, i in enumerate(Attributes):
    	for j in Top_Att:
        	if j in i:
            	position = n            	
	new_reach = sets[position]
	r = 0
	
	for i in range(0,len(dataframe.columns) - 1):
    	diff=0
    	for each_set in sets:
        	if len(each_set.difference(new_reach)) > diff:
            	diff = len(each_set.difference(new_reach))
            	set_to_add = sets.index(each_set)
        order.append(dataframe.columns[set_to_add])
    	new_reach = set.union(new_reach,sets[set_to_add])
        percentages.append(len(new_reach)/len(dataframe))
    	r=len(dataframe)
	
	return order,percentages,r
 
### Function for Graph 
def Result_Reach_Maxdiff(df, df_filter, no_attr, main_filter,second_filter):
	"""
	combination: Number of combinations
	df: Raw Scores
	df_filters: Filter data
	no_attr: Attribute shown per screen
	attr_filter: Filter values
	"""
	dataframe = filter_function(df, df_filter, main_filter,second_filter)
	
	attribute = df.shape[1]
	Result_Final=pd.DataFrame()
	Att_list = []
	Base = len(dataframe)
	for j in range(1,attribute+1):
    	comb=[]
    	comb = list(combinations(df.columns.values.tolist(),j))
    	if len(Att_list)>0:
        	res = [tup for tup in comb if all(i in tup for i in Att_list)]
    	else:
        	res = comb
    	database = dataframe.copy()
    	for items in res:
        	database[' , '.join(items)] = np.exp(database.loc[:, items].values).sum(1)/((np.exp(database.loc[:, items].values).sum(1)+(no_attr-1)))       
        	
    	database_ = database.copy()
    	if database_.shape[1]>attribute:
        	Result = database_.iloc[:,attribute:].mean(0)
    	else:
        	Result = database_.mean(0)
 
    	Result_Reach = pd.DataFrame(Result)
    	Result_Reach.reset_index(inplace = True)
    	Result_Reach.columns = ['Feature Combinations','Percentage Reach']
    	Result_Reach['Percentage Reach'] = Result_Reach['Percentage Reach']*100
    	Result_Reach['Percentage Reach'] = Result_Reach['Percentage Reach'].round(decimals=2)
        Result_Reach['Combination_No'] = j
    	Result_Reach['Base Size'] = Base
        Result_Reach['Combination_No'] = Result_Reach['Combination_No'].astype(np.uint8)
    	Result_Reach['Base Size'] = Result_Reach['Base Size'].astype(np.uint8)
    	Result_Reach_top = Result_Reach.nlargest(1,['Percentage Reach'])
    	if j>1:
        	new = Result_Reach_top["Feature Combinations"].str.split(" , ", n = j-1, expand = True)
        	new = new.transpose()
        	new.columns = ['list']
        	Att_list = (new['list']).tolist()
    	else:
       	Att_list =  (Result_Reach_top["Feature Combinations"]).tolist()
            	
    	Result_Final = Result_Reach_top.append(Result_Final, ignore_index=True)
 
	return Result_Final
 
### Function for Feature Name 
def remove_duplicate(df, df_filter, no_attr, main_filter,second_filter):
	"""
	combination: Number of combinations
	df: Raw Scores
	df_filters: Filter data
	no_attr: Attribute shown per screen
	attr_filter: Filter values
	"""	
	temp = Result_Reach_Maxdiff(df, df_filter, no_attr, main_filter,second_filter)
	temp = temp.sort_values('Percentage Reach')	
	New_lst = []
	New_2st = []
	temp['Feature']  = temp['Feature Combinations'].str.split(" , ")
	for i in temp['Feature']:
    	for j in i:
        	if j not in New_lst:
            	New_2st.append(j)
            	New_lst.append(j)
	
	temp['Feature']  = New_2st   
	return temp
 
### Function for Combination
def Result_Combination_Maxdiff(combination, df, df_filter, top, no_attr, main_filter,second_filter):
	"""
	combination: Number of combinations
	df: Raw Scores
	df_filters: Filter data
	top: Top number of combinations
	no_attr: Attribute shown per screen
	attr_filter: Filter values
	"""
	database = filter_function(df, df_filter, main_filter,second_filter)
	
	num_attr = database.shape[1]
	comb = list(combinations(database.columns.values.tolist(), combination))
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
    Result_Reach['Combination_No'] = Result_Reach['Combination_No'].astype(np.uint8)
	Result_Reach_top = Result_Reach.nlargest(top,['Percentage Reach'])
	return Result_Reach_top
 
 
result_df = remove_duplicate(df_Raw, df_filter, 4, 'All Respondents', 'All Respondents')
