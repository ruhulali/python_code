#BEN 

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:10:06 2020
 
@author: ruhul
"""
### ------ Error Search on Stackoverflow ------ ###
#import sys
#import webbrowser
#import urllib.parse
 
#def search_last_err_on_so():
#	last_err = sys.last_value
#	url = "https:stackoverflow.com/search?q=" + str(last_err)
#    webbrowser.open_new_tab(urllib.parse.quote(url))
	
### ------------------------------------------- ###
 
################################################################################################################
 
### QT2 --------------------------------------
### Import Packages	
import os
os.chdir('V:\\Projects\\2020\\Power BI\\BEN\\Materials\\Monthly_Data\\09-03-2021\\QT2')
 
import pandas as pd
 
### Import Current Month Data File
qt2 = pd.read_excel("QT2_QF2_F2_Subscription Link BI Report.xlsx")
 
### Data Check
qt2.head()
qt2.describe() # (include='all')
qt2.info()
qt2['value'].min()
qt2['base'].min()
 
 
### Check Columns
qt2['Country'].value_counts()
qt2['Year'].value_counts()
 
qt2['Month'].value_counts()
qt2['Month'] = qt2['Month'].str.replace('July C5', 'July')
 
qt2['Final Platform'].value_counts()
qt2['Final Platform'] = qt2['Final Platform'].replace(['Apple','Disney','Netflix-Film'],['Apple TV+','Disney+','Netflix-Film US'])
 
qt2['Platform Data'].value_counts()
 
qt2['Show/Film'].value_counts()
qt2['Show/Film'] = qt2['Film/Show'].str.replace('show', 'Show')
 
### Check Base and Value for Zero and remove
qt2['value'].describe()
qt2 = qt2[qt2.value != 0.0]
 
qt2['base'].describe()
qt2 = qt2[qt2.base != 0.0]
 
################################################################################################################
 
### Import QT2 & check columns
qt2_merged = pd.read_excel("D:\\KNOWLEDGE KORNER\\OFFICE\\BEN\\PowerBI\\Doc\\07-08-2020\\Link BI Subscription_Final.xlsx")
 
### Check columns
qt2_merged.info()
qt2.columns
qt2_merged.columns
 
### Insert "Sr No." columns
qt2.insert(0, 'Sr No.', 0)
 
### Change column names
if set(['Final Platform','Platform Data']).issubset(qt2.columns):
   qt2.rename(columns={"Final Platform": "Platform"}, inplace=True)
   qt2.rename(columns={"Platform Data": "BEN Data"}, inplace=True)
  
### Check for new column names  
if set(['Sr No.', 'Country', 'Year', 'Month', 'Platform', 'BEN Data','Show/Film',
	'variablename', 'response', 'base', 'value', 'percentage']).issubset(qt2.columns):
	print("File has NEW column names !!!")
else:
	print("File has OLD column names !!!")
 
### Append QT2 with QT2_Final
qt2_final = qt2_merged.append(qt2, ignore_index = True)
qt2_final.tail()
qt2_final.info()
 
### Fill "Sr No." Column
qt2_final['Sr No.'] = range(1, len(qt2_final) + 1)
qt2_final
 
################################################################################################################
 
### QT3 --------------------------------------
### Import Data File
os.chdir('V:\\Projects\\2020\\Power BI\\BEN\\Materials\\Monthly_Data\\09-03-2021\\QT3')
 
### Import Current Month Data File
qt3 = pd.read_excel("QT3_Link Bi Report.xlsx")
 
### Check Columns
qt3.head()
qt3.describe() # (include='all')
qt3.info()
qt3['value'].min()
qt3['base'].min()
 
 
qt3['Country'].value_counts()
qt3['Year'].value_counts()
 
qt3['Month'].value_counts()
qt3['Month'] = qt3['Month'].str.replace('October C5', 'October')
 
qt3['Final Platform'].value_counts()
qt3['Final Platform'] = qt3['Final Platform'].replace(['Apple','Disney','Netflix-Film'],['Apple TV+','Disney+','Netflix-Film US'])
 
qt3['Platform Data'].value_counts()
 
qt3.columns
qt3['Show/Film'].value_counts()
qt3['show/film'] = qt3['show/film'].str.replace('film', 'Film')
 
 
### Check Base and Value for Zero and remove
qt3['value'].describe()
qt3['value'].min()
qt3 = qt3[qt3.value != 0.0]
 
qt3['base'].describe()
qt3['base'].min()
qt3 = qt3[qt3.base != 0.0]
 
################################################################################################################
 
##-----------------------------------------------
 
### Import QT3 & check columns
qt3_merged = pd.read_excel("D:\\KNOWLEDGE KORNER\\OFFICE\\BEN\\PowerBI\\Doc\\07-08-2020\\QT3_LinkBI.xlsx")
 
qt3.info()
qt3_merged.info()
qt3.columns
qt3_merged.columns
 
 
### Change column names
if set(['Final Platform']).issubset(qt3.columns):
   qt3.rename(columns={"Final Platform": "Platform"}, inplace=True)
  
### Check for new column names  
if set(['Country', 'Year', 'Month', 'Platform', 'Platform Data','Show/Film',
	'variablename', 'response', 'base', 'value', 'percentage']).issubset(qt3.columns):
	print("File has NEW column names !!!")
else:
	print("File has OLD column names !!!")
 
 
### Append QT3 with QT3_Final
qt3_final = qt3_merged.append(qt3, ignore_index = True)
qt3_final.tail()
qt3_final.info()
 
################################################################################################################
 
### QF5 --------------------------------------------------
 
### Import Data File
os.chdir('V:\\Projects\\2020\\Power BI\\BEN\\Materials\\Monthly_Data\\09-03-2021\\QF5')
 
### Import Current Month Data File
qf5 = pd.read_excel("QF5_Link Bi Report.xlsx")
 
### Check Columns
qf5.info()
qf5.head()
qf5.describe() # (include='all')
qf5['value'].min()
qf5['base'].min()
 
 
qf5['Country'].value_counts()
qf5['Year'].value_counts()
 
qf5['Month'].value_counts()
qf5['Month'] = qf5['Month'].str.replace('October C5', 'October')
 
qf5['Final Platform'].value_counts()
qf5['Final Platform'] = qf5['Final Platform'].replace('Netflix-Film','Netflix-Film US')
 
qf5['Platform Data'].value_counts()
 
 
### Check Base and Value for Zero and remove
qf5['value'].describe()
qf5['value'].min()
qf5 = qf5[qf5.value != 0.0]
 
qf5['base'].describe()
qf5['base'].min()
qf5 = qf5[qf5.base != 0.0]
 
################################################################################################################
 
### Check Final Merged Files --------------------------------
qt2_final.info()
qt3_final.info()
 
qt2_final.head()
qt3_final.head()
 
qt2_final.describe()
qt3_final.describe()
 
### Export final files
qt2_final.to_excel (r'D:\\KNOWLEDGE KORNER\\OFFICE\\BEN\\PowerBI\\Doc\\07-08-2020\\Output_qt2.xlsx', index = False, header=True)
qt3_final.to_excel (r'D:\\KNOWLEDGE KORNER\\OFFICE\\BEN\\PowerBI\\Doc\\07-08-2020\\Output_qt3.xlsx', index = False, header=True)
 
################################################################################################################
 
qt2.to_excel (r'Output_qt2.xlsx', index = False, header=True)
qt3.to_excel (r'Output_qt3.xlsx', index = False, header=True)
qf5.to_excel (r'Output_qf5.xlsx', index = False, header=True)


################################################################################################################

Demo Data Manipulation

FINAL_PLATFORM_val - Platform name change
Attribute_Vars - T2 - Show
                 F2 - Film
                 
