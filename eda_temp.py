# -*- coding: utf-8 -*-
"""
Python Codes
 
"""
### https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf
 
#==============================================================================
### Update Conda and Anaconda
#conda update --all
#conda update conda
#conda update anaconda
#conda update anaconda navigator
 
### Get and Change Path
import os
path="E:\\A_NOTES\\Analytics Notes\\Python\\Python_Syntax"
os.chdir(path)
os.getcwd()
 
### Checking Version of package
from sklearn import __version__
print(__version__)
print ('Matplotlib version: ', mpl.__version__)
 
#==============================================================================
### Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid") # darkgrid, whitegrid, dark, white, ticks
sns.set(rc={'figure.figsize':(10,5)})
plt.style.use('ggplot')
# (bmh, fivethirtyeight, seaborn-dark, ggplot)
# ('ggplot', 'seaborn-bright', 'seaborn-ticks', 'seaborn-talk', 'seaborn-muted', 'dark_background', 'tableau-colorblind10',
#  'fast', 'seaborn-white', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-paper', 'seaborn', 'bmh', 'fivethirtyeight',
#  'Solarize_Light2', 'seaborn-notebook', 'classic', 'seaborn-poster', 'seaborn-pastel', 'seaborn-dark-palette',
#  'seaborn-deep', '_classic_test', 'seaborn-whitegrid', 'grayscale', 'seaborn-darkgrid']
 
sns.get_dataset_names()
sns.load_dataset("tips") # titanic
 
### For Jupyter Notebook
# import plotly.offline as pyo
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import cufflinks as cf
# %matplotlib inline
# %matplotlib notebook
 
 
### For Colab
# Import Files
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('01-01-2020-TO-27-08-2020RELIANCEEQN.csv')
 
# Export Files
from google.colab import files
files.download("Filename.xlsx")
 
 
### Pandas ProfileReport
#%matplotlib inline
import pandas_profiling
from pandas_profiling import ProfileReport
#profile = ProfileReport(df, title="Tips Dataset", html={'style': {'full_width': True}}, sort="None")
profile = ProfileReport(df) # for large dataset (minimal=True)
profile.to_notebook_iframe()
#profile.to_file(output_file="your_report.html")
 
# For Jupyter Notebook
from pandas_profiling import ProfileReport
profile = ProfileReport(df, explorative=True, minimal = True)
profile.to_file('output.html')
 
### Dtale (EDA)
# For Spyder
import dtale
import plotly.express as px
d = dtale.show(df, ignore_duplicate=True)
d.open_browser()

# For Jupyter Notebook
import dtale
dtale.show(df)

 
### Pandas_ui
import seaborn as sns
df = sns.load_dataset("tips")
from pandas_ui import *
pandas_ui('D:\\KNOWLEDGE KORNER\\ANALYTICS\\MISC\\Practice\\Kaggle & Hackathons\\Tips\\tips.csv')
 
#==============================================================================
### Upload CSV Files
# tips = pd.read_csv("Train.csv")
# tips = pd.read_csv("D:/KNOWLEDGE KORNER/ANALYTICS/MISC/NOTES/Analytics Notes/Practice/Kaggle & Hackathons/Tips/tips.csv")
# tips = pd.read_csv("E:/A_NOTES/Analytics Notes/Practice/Kaggle & Hackathons/Tips/tips.csv")
sns.get_dataset_names() #get all the datasets under sns
tips = sns.load_dataset("tips")  # Load Tips Dataset
iris = sns.load_dataset('iris') # Load Iris Dataset
flights = sns.load_dataset('flights') # Load Flights Dataset
 
### Save in CSV
tips.to_csv(r'tips.csv')
tips.to_csv(r"D:\Ruhul_Data\Drive_Data_Ruhul\A_NOTES\Analytics Notes\Python\Data Analysis with Python-Cognitive Classes\Data\automobile.csv")
 
### Upload HTML Files
listofplayers = pd.read_html('https://en.wikipedia.org/wiki/World_Soccer_(magazine)')
listofplayers[0]
 
 
#==============================================================================
### Data Info
tips.head()
tips.columns
tips.info()
tips.nunique()
tips.skew()
print("Skewness: %f" % tips['total_bill'].skew())
print("Kurtosis: %f" % tips['total_bill'].kurt())
tips.describe(include='all') ## .describe().transpose(), include=['object']
tips["total_bill"].describe()
tips["total_bill"].idxmax()
tips.at[170, 'total_bill']
 
# Reset Index to new column
wq.reset_index(drop=False, inplace=True)
wq.set_index('Date', inplace=True)
 
# Rename Column Name
tips.rename(columns={"old": "new"}, inplace=True)
 
# Make all column labels of type string
tips.columns = list(map(str, tips.columns))
 
# Column Total (Numerical)
tips['Total'] = tips.sum(axis=1)
 
# Selecting full row data (all columns)
print(df_can.loc['Japan'])
 
# Aalternate methods
print(df_can.iloc[87])
print(df_can[df_can.index == 'Japan'].T.squeeze())
 
# Selecting more than 2 columns
print(df_can.loc['Japan', 2013])
 
# Pass mutliple criteria in the same line.
df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]
 
#==============================================================================
### Missing Value
tips.isnull().values.any()
tips.isnull().sum()
msg = sns.heatmap(tips.isnull(), cbar=True, yticklabels=False, cmap='viridis');
msg.set_ylim(sorted(msg.get_xlim(), reverse=True))
 
total = tips.isnull().sum().sort_values(ascending=False)
percent = (tips.isnull().sum()/tips.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
round(missing_data.head(20)*100,2)
 
## Dropping Missing Values
tips.dropna(subset=["tip"], axis=0, inplace = True)
 
##   Missing Values
mean = tips["tips"].mean()
tips["tips"].replace(np.nan, mean)
 
#==============================================================================
###Data Normalization
## Simple Feature Scaling
df["length"] = df["length"]/df["length"].max()
 
## Min-Max
df["length"] = (df["length"]-df["length"].min())/(df["length"].max()-(df["length"].min())
            	
## Z-Score
df["length"] = (df["length"]-df["length"].mean())/df["length"].std()
 
#==============================================================================
### Select all numerical & categorical variables
tips_num = tips.select_dtypes(include=['float64', 'int64'])
tips_cat = tips.select_dtypes(include=['object'])
 
### Change data type
tips["size"]= tips["size"].astype(object)
 
#==============================================================================
### Built-In Data Viz (Quick Charts)
tips['total_bill'].plot.hist(bins=40)
tips['total_bill'].plot.area(alpha=0.4)
tips['total_bill'].plot.bar() #stacked=True
tips.plot.line('total_bill', 'tip', figsize=(12,3), lw=1) #best for timeseries data
tips.plot.scatter('total_bill', 'tip', c='size', cmap='coolwarm') #s=tips['size']*10
tips['total_bill'].plot.box()
tips.plot.hexbin('total_bill', 'tip', gridsize=20, cmap='coolwarm')
 
 
### Numerical Distribution Plots (Histogram, JointPlot, DistPlot & PairPlot)
sns.distplot(tips.tip, bins=40)
sns.distplot(tips.total_bill, bins=40)
sns.jointplot('total_bill', 'tip', data=tips, kind='reg'); #kind='reg' (regression line), alpha=0.5
tips.hist('tip',by='sex', bins=40); #kde=True (kernal density estimation)
tips_num.hist(bins=40)
tips.hist()
sns.FacetGrid(tips, 'sex', 'time').map(plt.hist, 'tip', edgecolor='w') #sns.distplot, plt.scatter
sns.FacetGrid(tips, 'sex', 'time').map(plt.scatter, 'total_bill', 'tip', edgecolor='w').add_legend()
sns.PairGrid(iris).map(plt.scatter)
sns.PairGrid(iris).map_diag(sns.distplot).map_upper(plt.scatter).map_lower(sns.swarmplot) #kdeplot, barplot, swarmplot
sns.pairplot(tips, hue='sex')
for i in range(0, len(tips_num.columns),5):
	sns.pairplot(tips_num, y_vars=['tip'], x_vars=tips_num.columns[i:i+5])
 
 
### Categorical Distribution Plots (Bar Chart, Count Plot )
sns.barplot('day', 'tip', hue='sex', data= tips) #estimator=np.std, hue='day'
sns.countplot('sex', data=tips)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # To place legend outside the chart
sns.catplot('sex', col='time', hue= 'smoker', kind='count', data=tips)
sns.boxplot('day','tip', hue='sex', data=tips)
sns.violinplot('day', 'total_bill', data=tips, hue='sex', split=True)
sns.swarmplot('day', 'total_bill', data= tips)
# Can use both Violin and Swarm for better look
# sns.violinplot('day', 'total_bill', data= tips)
# sns.swarmplot('day', 'total_bill', data= tips, color='black')
sns.factorplot('day', 'total_bill', data=tips, kind='bar') #kind='violin', 'bar', 'box', 'swarm'
 
 
### Matrix Plot (Correlation)
tips.corr()
hmap = sns.heatmap(tips.corr(), annot=True)
hmap.set_ylim(sorted(hmap.get_xlim(), reverse=True))
pearson_coef, p_value = stats.pearsonr(tips['tip', tips['sex'])
 
fp = flights.pivot_table(index='month', columns='year', values= 'passengers')
sns.heatmap(fp)
sns.clustermap(fp) #standard_scale=1
 
 
### Regression Plot (Scatter, LmPlot)
tips.plot.scatter('total_bill', 'tip')
sns.regplot('total_bill', 'tip', data=tips)
sns.residplot(tips['total_bill'], tips['tip'])
sns.lmplot(x='total_bill', y='tip', hue='sex', data=tips, scatter_kws={'edgecolors': 'w'}, fit_reg=True) #scatter_kws={'s'=50}, markers=['o','v'], size=5
 
 
#==============================================================================
### Plotly & Cufflinks
init_notebook_mode(connected=True)
cf.go_offline()
 
# Create Sample Data
dt = pd.DataFrame(np.random.randn(100,4), columns='A B C D'.split())
dt2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values':[32,43,50]})
dt.iplot()
 
# Scatter Plot
dt.iplot(kind='scatter', x='A', y='B', mode='markers', size=15)
tips.iplot(kind='scatter', x='total_bill', y='tip', mode='markers', size=15)
# For Scatter Plotly Plot in Spyder
import plotly.graph_objects as go
from plotly.offline import plot
trace = go.Scatter(x=tips['total_bill'], y=tips['tip'], mode='markers')
data = [trace]
plot(data)
 
# Bar Plot
dt2.iplot(kind='bar', x='Category', y='Values')
tips.count().iplot(kind='bar') #sum
 
# Box Plot
dt.iplot(kind='box')
tips.iplot(kind='box')
 
# Surface Plot
dt3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[500,400,300,200,100]})
dt3.iplot(kind='surface', colorscale='rdylbu')
 
# Histogram
dt.iplot(kind='hist')
tips.iplot(kind='hist')
# For Histogram Plotly Plot in Spyder
import plotly.graph_objects as go
from plotly.offline import plot
trace = go.Histogram(x=tips['day'], histfunc='sum') #histfunc='sum','count'  #histnorm='percent','probability'
data = [trace]
plot(data)
 
# Spread
dt[['A','B']].iplot(kind='spread')
tips[['total_bill','tip']].iplot(kind='spread')
 
# Bubble
dt.iplot(kind='bubble', x='A', y='B', size='C')
tips.iplot(kind='bubble', x='total_bill', y='tip', size='size')
 
# Scatter
dt.scatter_matrix()
iris.scatter_matrix()
 
#==============================================================================
### Categorical Columns Summarized
def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='viridis', verbose=True):
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
categorical_summarized(tips, x = 'day')
categorical_summarized(tips, x = 'time', hue='sex')
 
 
### Numerical Columns Summarized
def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='viridis', ax=None, verbose=True, swarm=True):
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
quantitative_summarized(tips, y = 'tip')
quantitative_summarized(tips, y = 'tip', x = 'sex', swarm=False)
quantitative_summarized(tips, y = 'tip', x = 'sex', hue = 'day', swarm=False)
 
 
#==============================================================================
### Frequencies and Crosstabs
tips['sex'].value_counts() # value_counts, unique, nunique
tips.groupby('day').count() # sum, mean, describe, count
pd.value_counts(tips.day).to_frame().reset_index() # counts
pd.crosstab(tips.sex, tips.day, margins=True) # crosstab counts
pd.crosstab(tips.sex, tips.day, normalize='columns')*100 # column %
pd.crosstab(tips.sex, tips.day, normalize='index')*100 # row %
tips.pivot_table(values=["tip"], index=["sex", "smoker", "time"], aggfunc=np.mean) # pivot table (numerial as x)
sns.heatmap(tips.pivot_table(values=["tip"], index=["sex", "time"], aggfunc=np.mean), annot=True, cbar=True)
ctab = sns.heatmap(pd.crosstab([tips.sex], [tips.day]), cmap="tab10", annot=True, cbar=True) # visualised crosstab
ctab.set_ylim(sorted(ctab.get_xlim(), reverse=True))
 
plt.figure(figsize=(8,4))
splot = sns.barplot(data=tips, x = 'sex', y = 'total_bill', ci = None)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height() / 2),
	ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
 
plt.figure(figsize=(8,4))
groupedvalues=tips.groupby('day').mean().reset_index() #sum, count, mean
g=sns.barplot(x='day',y='total_bill',data=groupedvalues)
for index, row in groupedvalues.iterrows():
	g.text(row.name,row.tip, round(row.total_bill,2), color='black', ha="center", bbox=dict(facecolor='white', alpha=5.0))
	
 
#==============================================================================
### Linear Regression
 
# Simple
from sklearn import linear_model
regr = linear_model.LinearRegression() #create linear regression object
y = tips['tip']
x = tips[['size','total_bill']]
regr.fit(x,y) #train the model using training model
regr.coef_
regr.intercept_
#tip predicted  = 0.66 + 0.19 * size + 0.09 * total_bill
 
 
# Detailed
# Create Model
ec.columns
y = ec['Yearly Amount Spent']
X = ec[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
 
# Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
 
# Train Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
 
print('Intercept =', lm.intercept_)
print('Coefficient =', lm.coef_)
print('R^2 =', lm.score) #R^2 value
 
# Create DataFrame of Coeff
X_train.columns
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
cdf
 
# Prediction
y_pred = lm.predict(X_test)
y_pred #predicted values
y_test #actual values
lm.predict_proba(X_test)
 
# Random Prediction, y=mx+c
25.98154972*30 + 38.59015875*10 + 0.19040528*25 + 61.27909654*2 + -1047.9327822502387
lm.predict([[30,10,25,2]])
 
# Plot Model
plt.scatter(y_test, y_pred)
sns.regplot('total_bill', 'tip', data=tips)
 
ax1 = sns.distplot(tips['tip'], hist=False, color='r', label='Actual')
sns.distplot(y_pred, hist=False, color='b', label='Predicted', ax=ax1)
 
# Plot Residuals
sns.distplot((y_test-y_pred), bins=40)
sns.residplot(tips['total_bill'], tips['tip'])
 
# Evaluation Metrics
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred) #MAE
metrics.mean_squared_error(y_test, y_pred) #MSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred)) #RMSE
 
from sklearn import metrics
print('MAE -', metrics.mean_absolute_error(y_test, y_pred))
print('MSE -', metrics.mean_squared_error(y_test, y_pred))
print('RMSE -', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared -', metrics.explained_variance_score(y_test, y_pred)*100)
## R^2 shows how close the data is to the predicted reg line
 
#==============================================================================
### logistic Regression
 
# Create Model
y = train['Survived']
X = train.drop('Survived', axis=1)
 
# Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
 
# Train Model
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, y_train) # Train Model
lm.score(X_test, y_test) # Get initial Scores
 
# Predict Model
y_pred = lm.predict(X_test) # Prediction
y_pred #predicted values
y_test #actual values
lm.predict_proba(X_test)
 
# Evaluate Model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
 
# Plot Model
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=True, fmt=".10g")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 
# Accuracy: Fraction predicted correctly- TP+TN/ TP+TN+FP+FN
# Recall/Sensitivity/True Positive Rate: Fraction of recalling actual positive as predicted positive- TP/ (TP+FN)
# Precision: Fraction of precisely predicted positive as actual positive- TP/ (TP+FP)
# F1-Score: Harmonic mean of both Recall and Precision- 2*((precision*recall)/(precision+recall)).
        	# As a rule of thumb, the weighted average of F1 should be used to compare classifier models
# Support: It is the number of occurence of the given class in your dataset
 
#==============================================================================
### KNN
 
# Standardize Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cd.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(cd.drop('TARGET CLASS', axis=1))
scaled_features
cd_feat = pd.DataFrame(scaled_features, columns = cd.columns[:-1])
cd_feat.head()
 
### Create Model
y = cd['TARGET CLASS']
X = cd_feat
 
# Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
 
# Train Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
 
# Predict Model
y_pred = knn.predict(X_test)
 
# Evaluate Model
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
# Elbow Method
error_rate = []
for i in range(1,40):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train, y_train)
	y_pred_i = knn.predict(X_test)
    error_rate.append(np.mean(y_pred_i != y_test))
 
error_rate
 
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
 
# Re-Train Model
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)	
 
# Re-Evaluate Model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
#==============================================================================
### Decision Tree
 
# Create Model
final_loan.columns
y = final_loan['not.fully.paid']
X = final_loan.drop('not.fully.paid', axis=1)
 
# Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
 
# Train Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
 
# Prediction
y_pred = dtree.predict(X_test)
 
# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
 
#==============================================================================
### Random Forest
 
# Create Model
final_loan.columns
y = final_loan['not.fully.paid']
X = final_loan.drop('not.fully.paid', axis=1)
 
# Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
 
# Train Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
 
# Prediction
y_pred = rfc.predict(X_test)
 
# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
#==============================================================================
### SVM
 
# Create Model
iris.columns
y = iris['species']
X = iris.drop('species', axis=1)
 
# Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
 
# Train Model
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
 
# Prediction
y_pred = svm.predict(X_test)
 
# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
# Grid Search CV
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train, y_train)
print('Best Param -', grid.best_params_)
print('Best Estimator -','\n', grid.best_estimator_)
print('Best Score -', grid.best_score_)
 
# Prediction
grid_pred = grid.predict(X_test)
 
# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
#==============================================================================
### KMeans
 
# Train Model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(college.drop('Private', axis=1))
 
kmeans.cluster_centers_
kmeans.labels_
 
# Evaluate Model
def convert(private):
	if private == 'Yes':
    	return 1
	else:
    	return 0
 
college['Cluster'] = college['Private'].apply(convert)
college.head()
college.info()
 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(college['Cluster'], kmeans.labels_))
print(classification_report(college['Cluster'], kmeans.labels_))
 
#==============================================================================
### PCA
 
### Import Data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])
 
cd = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
cd.info()
 
### Standardize Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cd)
scaled_data = scaler.transform(cd)
 
# Train Model
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
 
# Plot Model
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='rainbow')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
 
df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
df_comp.head()
sns.heatmap(df_comp)
 
#==============================================================================
### Recommender System
 
### Import Data
column_name = ['user_id', 'item_id', 'rating', 'timestamp']
 
df = pd.read_csv("D:\\KNOWLEDGE KORNER\\ANALYTICS\\Udemy\\Python for DS & ML Bootcamp\\Refactored_Py_DS_ML_Bootcamp-master\\19-Recommender-Systems\\u.data",
             	sep='\t', names=column_name)
 
df.head()
df.info()
 
movie_titles = pd.read_csv("D:\\KNOWLEDGE KORNER\\ANALYTICS\\Udemy\\Python for DS & ML Bootcamp\\Refactored_Py_DS_ML_Bootcamp-master\\19-Recommender-Systems\\Movie_Id_Titles.csv")
movie_titles.head()
movie_titles.info()
 
df = pd.merge(df, movie_titles, on='item_id')
df.head()
df.info()
 
### EDA
# Mean ratings of movies sorted desc
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
# Count of most rated movie sorted desc
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
# Create DataFrame with mean rating
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
# Create column having ratings per movies
ratings['no. of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()
 
# Visualize DataFrame
sns.distplot(ratings['no. of ratings'], bins=80)
sns.distplot(ratings['rating'], bins=80)
sns.jointplot('rating', 'no. of ratings', data=ratings, alpha=0.5)
 
# Create Matrix
df.head()
movie_mat = df.pivot_table(index='user_id', columns='title', values='rating')
movie_mat.head()
ratings.sort_values('no. of ratings', ascending=False).head()
 
# Grab Movie Ratings
star_wars_user_rating = movie_mat['Star Wars (1977)']
liar_liar_user_rating = movie_mat['Liar Liar (1997)']
star_wars_user_rating.head()
 
# Check Correlation of Starwar
similar_to_starwar = movie_mat.corrwith(star_wars_user_rating)
corr_starwars = pd.DataFrame(similar_to_starwar, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['no. of ratings'])
corr_starwars[corr_starwars['no. of ratings']>100].sort_values('Correlation', ascending=False).head()
 
# Check Correlation of Liar Liar
similar_to_liarliar = movie_mat.corrwith(liar_liar_user_rating)
corr_liar_liar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liar_liar.dropna(inplace=True)
corr_liar_liar = corr_liar_liar.join(ratings['no. of ratings'])
corr_liar_liar[corr_liar_liar['no. of ratings']>100].sort_values('Correlation', ascending=False).head()
 
#==============================================================================
 
