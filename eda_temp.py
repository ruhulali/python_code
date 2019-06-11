# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
### https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
sns.set_style("whitegrid")
%matplotlib inline
plt.style.use('ggplot') # (bmh, fivethirtyeight, seaborn-dark, ggplot)


### Upload Files 
df = pd.read_csv("Train.csv")
df = pd.read_csv("D:/KNOWLEDGE KORNER/ANALYTICS/MISC/NOTES/Analytics Notes/Practice/Kaggle & Hackathons/Tips/tips.csv")
df = pd.read_csv("E:/A_NOTES/Analytics Notes/Practice/Kaggle & Hackathons/Tips/tips.csv") 
df = sns.load_dataset("tips")  # Load data


### Data Info
df.info()
df.head(10)
df.describe()
df.nunique()


### Change data type
df["size"]= df["size"].astype(object) 


### Missing Value
df.isnull().values.any()
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')  
# or
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)*100


### Select all numerical & categorical variables
tips_num = df.select_dtypes(include=['float64', 'int64'])
tips_cat = df.select_dtypes(include=['object'])
sns.pairplot(df, hue='sex')
for i in range(0, len(tips_num.columns),5):
    sns.pairplot(tips_num, y_vars=['tip'], x_vars=tips_num.columns[i:i+5])    

### Basic 
df['sex'].value_counts() # value_counts, unique, nunique
crosstab = df.groupby('sex')
crosstab.describe() # sum, mean, describe

### Correlation 
plt.figure()
sns.heatmap(df.corr(),cmap='Greens',annot=False)
# or
k = 12
cols = df.corr().nlargest(k, 'tip')['tip'].index
cm = df[cols].corr()
plt.figure()
sns.heatmap(cm, annot=True, cmap = 'viridis')

### Bar Chart 
sns.countplot(x='sex', data=df)
sns.factorplot(x='sex', col='day', kind='count', data=df)
sns.barplot(x="time", y="total_bill", hue="sex", data=df)

### Scatter Plot
sns.lmplot(x='total_bill', y='tip', hue='sex', data=df, scatter_kws={'edgecolors': 'w'}, fit_reg=True)
plt.show()

### Histogram
num_bins = 10
plt.hist(df['tip'], num_bins, normed=1, facecolor='blue', alpha=0.5)
# or
sns.distplot(df.tip, kde=True)
# or
df.hist(column="tip", by="sex",bins=10)
#or
tips_num.hist(bins=10)

### Box Plot
y = list(df.tip) 
plt.boxplot(y)     
# or
df.boxplot(column="tip",by="sex")
# or
sns.boxplot(x="day", y="tip", data=df)    


### Categorical Summarized
def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()
#
categorical_summarized(df, y = 'day')
categorical_summarized(df, y = 'day', hue='sex')


### Quantitative Summarized
def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    plt.show()    
#
quantitative_summarized(df, y = 'tip', verbose=False, swarm=True)
quantitative_summarized(df, y = 'tip', x = 'sex', verbose=True, swarm=True)
quantitative_summarized(df, y = 'tip', x = 'sex', hue = 'day', verbose=False, swarm=False)


### Frequencies and Crosstabs
pd.value_counts(df.day).to_frame().reset_index() # counts
pd.crosstab(df["sex"], df["day"], margins=True) # crosstab counts
pd.crosstab(df.sex, df.day, normalize='columns')*100 # column %
pd.crosstab(df.sex, df.day, normalize='index')*100 # row %
sns.heatmap(pd.crosstab([df.sex], [df.day]), cmap="tab10", annot=True, cbar=True) # visualised crosstab
df.pivot_table(values=["tip"], index=["sex", "smoker", "time", "size"], aggfunc=np.mean) # pivot table (numerial as x)
sns.heatmap(df.pivot_table(values=["tip"], index=["sex", "time", "smoker"], aggfunc=np.mean), annot=True, cbar=True)

plt.figure()
splot = sns.barplot(data=df, x = 'sex', y = 'total_bill', ci = None)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

groupedvalues=df.groupby('time').sum().reset_index()
g=sns.barplot(x='time',y='tip',data=groupedvalues)
for index, row in groupedvalues.iterrows():
    g.text(row.name,row.tip, round(row.total_bill,2), color='black', ha="center", bbox=dict(facecolor='white', alpha=5.0))

#####################

 




