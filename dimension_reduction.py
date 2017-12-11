#perform pca to get best k features
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import math
dataframe = pd.read_csv('./2012/combinedData2016.csv')
dataframe = dataframe.replace('N',0)
dataframe = dataframe.replace('-',0)
lb = LabelEncoder()
scaler = StandardScaler()
X = [a[1:] for a in dataframe[dataframe.columns[:-1]].values]
temp = np.nan_to_num(X)

for i in range(len(temp)):
	for j in range(len(temp[0])):
		if type(X[i][j]) == type('NaN') :
			temp[i][j] = 0.0
		if np.isnan(temp[i,j]):
			temp[i][j] = 0.0
X = temp.copy().astype(float)
print(np.max(X))
Y = dataframe[dataframe.columns[-1]].values
X = scaler.fit_transform(X)
k = 50

pca = PCA(n_components = k)
pca.fit(X)
var = pca.explained_variance_ratio_
plt.plot(var)
plt.show()
