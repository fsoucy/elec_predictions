import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
import pdb
import get_y 
import sklearn.linear_model
import combineData as cd 




folder = './rem_data/'
fnames = ['population.csv']

X = cd.loadFilesFrom(folder)
y = pd.read_csv('./election_results.csv')
X = cd.addYcol(X,y)

mat = X.as_matrix()
mat = list(mat)
for i in range(len(mat)):
    for j in range(len(mat[i])):
        val = mat[i][j]
        try:
            mat[i][j] = float(val)
        except:
            mat[i][j] = 0.0
mat = np.array(mat)
np.random.shuffle(mat)
X = mat[:,:-1]
X = X.T
np.random.shuffle(X)
X = X.T
X = X[:,:250]
Y = mat[:,-1]
trainX = X[:600,:]; trainY = Y[:600];
testX = X[600:, :]; testY = Y[600:];


model = sklearn.linear_model.LinearRegression(fit_intercept=True)
pdb.set_trace()
model.fit(trainX,trainY)
print("Train R^2:" + str(model.score(trainX, trainY)))
print("Test R^2:" + str(model.score(testX,testY)))

num_correct = 0
total = 0
for i in range(trainX.shape[0]):
    x = np.array([trainX[i,:]]); y = trainY[i]; predictY = model.predict(x)[0]
    if y < 0.5:
        if predictY < 0.5:
            num_correct += 1
    if y > 0.5:
        if predictY > 0.5:
            num_correct += 1
    total += 1
print("Acc:" + str(num_correct/total))
