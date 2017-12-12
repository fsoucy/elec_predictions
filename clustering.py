import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
import pdb
import get_y 
import sklearn.linear_model
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture as GM
import combineData as cd 
import copy
import matplotlib.pyplot as plt

fname = './cleanedData/'

X = cd.loadFilesFrom(fname)
Y = pd.read_csv('./election_results.csv')

X = cd.addYcol(X,Y)

df = X.copy(deep=True)


Y = X.iloc[:,-1].as_matrix()
X = X.iloc[:,:-1].as_matrix()

temp = np.nan_to_num(X)
for i in range(len(temp)):
	for j in range(len(temp[0])):
		if type(X[i,j]) == type('NaN') :
			temp[i,j] = 0.0
		if np.isnan(temp[i,j]):
			temp[i,j] = 0.0


X = temp.copy().astype(float)



ks = [3]
for k in ks:
	model = GM(k,reg_covar = 1e-3)
	model.fit(X,Y)
	# print(model.predict(X))
d = {}
Y = Y.reshape(Y.size,1)
pred = model.predict(X)
for k in range(ks[0]):
	d[k] = []
print(pred)
for i in range(len(pred)):
	# print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	temp = d.get(pred[i],[])
	temp.append(Y[i])
	d[pred[i]] = temp
for k,v in d.items():
	print('##################################')
	print('Cluster:',k)
	arr = np.concatenate(v).reshape(len(v),1)
	tot = 0
	for i in range(len(arr)):
		if type(arr[i,0]) == type(None):
			# print('ERROR!', i)
			continue
		tot += arr[i,0]
	avg = tot/arr.size
	tot = 0
	for i in range(len(arr)):
		if type(arr[i,0]) == type(None):
			# print('ERROR!',i)
			continue
		tot += (avg - arr[i,0])**2
	if arr.size > 1:
		std = (tot/(arr.size -1))**0.5
	else:
		std = 0
	print('Percent GOP Mean:',avg)
	print('Percent GOP STD:',std)
	print('Number Assigned:',arr.size)

print('columsn:',df.columns)
plt.scatter(X[:,0],X[:,1],c=pred)
plt.legend()
plt.show()