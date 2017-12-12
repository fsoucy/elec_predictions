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

fname = './cleanedData/'

X = cd.loadFilesFrom(fname)
Y = pd.read_csv('./election_results.csv')

X = cd.addYcol(X,Y)


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

k = 2
model = GM(k,reg_covar = 1e-4)
model.fit(X,Y)
