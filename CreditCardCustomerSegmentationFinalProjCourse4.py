import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

data_original = pd.read_csv('D:/Uni/Masters/Python/IBMMachineLearning/Course 4/FinalProj/CC GENERAL.csv') 
data_original.isnull().sum()
data_original = data_original.dropna(axis=0)
data_original.shape

data_original = data_original.drop(['CUST_ID'], axis = 1)
data_original.shape

data = data_original.copy()
data.head().T

data.shape
data.head()

corr_mat = data.corr()
for x in range(len(corr_mat)): 
	corr_mat.iloc[x,x] = 0
	
corr_mat
plt.figure(figsize=(12,12))
sns.heatmap(corr_mat, annot = True)

corr_mat.abs().max().sort_values(ascending=False)
(corr_mat.abs().max().sort_values(ascending=False) > 0.79).sum()
corr_mat.abs().idxmax()

#Skew 
skew_cols = data.skew().sort_values(ascending = False) 
skew_cols = skew_cols.loc[skew_cols > 0.75]
print(skew_cols)
for col in skew_cols.index.tolist(): 
	data[col] = np.log1p(data[col])

s = StandardScaler() 
data = s.fit_transform(data) 

data = pd.DataFrame(data, columns = data_original.columns)
data

pca_list = list() 
feature_weight_list = list() 

for n in range(1,18): 
	PCAmod = PCA(n_components = n) 
	PCAmod.fit(data) 
	
	pca_list.append(pd.Series({'n': n, 'model': PCAmod, 'var': PCAmod.explained_variance_ratio_.sum()}))
	weights = PCAmod.explained_variance_ratio_.reshape(-1,1)/PCAmod.explained_variance_ratio_.sum() 
	overall_contributions = np.abs(PCAmod.components_)*weights 
	abs_feature_values = overall_contributions.sum(axis =0) 	
	feature_weight_list.append(pd.DataFrame({'n':n, 'features': data.columns, 'values': abs_feature_values/(abs_feature_values.sum())}))
	
pca_df = pd.concat(pca_list, axis = 1).T.set_index('n') 
print(pca_df)

features_df = pd.concat(feature_weight_list).pivot(index = 'n', columns = 'features', values = 'values')
features_df.T

PCAmoduse = PCA(n_components=11)
data_new = PCAmoduse.fit_transform(data) 
data_new = pd.DataFrame(data_new, columns = ['col' + str(x) for x in range(1,12)])
data_new

km = KMeans(n_clusters=3) 
y_pred_km = km.fit_predict(data_new)
data_new['Clusters']=y_pred_km
#Add it also to the original data frame 
data['Clusters']=y_pred_km

sns.countplot(x=data['Clusters'])
ax = sns.scatterplot(data = data, x= data['BALANCE'], y=data['PURCHASES'], hue = data['Clusters'])
ax = sns.scatterplot(data = data, x= data['PURCHASES'], y=data['CREDIT_LIMIT'], hue = data['Clusters'])
ax = sns.scatterplot(data = data, x= data['CREDIT_LIMIT'], y=data['PURCHASES'], hue = data['Clusters'])
ax = sns.scatterplot(data = data, x= data['BALANCE'], y=data['CREDIT_LIMIT'], hue = data['Clusters'])
sns.pairplot(data, hue = 'Clusters')
AC = AgglomerativeClustering(n_clusters=3) 
y_pred_ac = AC.fit_predict(data_new)
data_new['Clusters']=y_pred_ac


data['Clusters']=y_pred_ac
sns.pairplot(data, hue = 'Clusters')
