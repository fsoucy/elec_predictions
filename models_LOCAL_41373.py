import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
import pdb
import get_y 
import sklearn.linear_model
from sklearn import preprocessing
import combineData as cd 




folder = './cleanedData/'

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
if (X.shape[1] > 180):
    X = X[:,:180]
print("Num features: " + str(X.shape[1]))
Y = mat[:,-1]
trainX = X[:600,:]; trainY = Y[:600];
testX = X[600:, :]; testY = Y[600:];


model = sklearn.linear_model.LinearRegression(fit_intercept=True)
model.fit(trainX,trainY)
print("Train R^2:" + str(model.score(trainX, trainY)))
print("Test R^2:" + str(model.score(testX,testY)))

def evaluateAccuracy(model, X, Y, dataType):
    num_correct = 0
    total = 0
    for i in range(X.shape[0]):
        x = np.array([X[i,:]]); y = Y[i]; predictY = model.predict(x)[0]
        if y < 0.5:
            if predictY < 0.5:
                num_correct += 1
        if y > 0.5:
            if predictY > 0.5:
                num_correct += 1
        total += 1
    print(dataType + " Acc:" + str(num_correct/total))
evaluateAccuracy(model, trainX, trainY, "Training")
evaluateAccuracy(model, testX, testY, "Test")



# num_correct = 0
# total = 0
# for i in range(trainX.shape[0]):
#     x = np.array([trainX[i,:]]); y = trainY[i]; predictY = model.predict(x)[0]
#     if y < 0.5:
#         if predictY < 0.5:
#             num_correct += 1
#     if y > 0.5:
#         if predictY > 0.5:
#             num_correct += 1
#     total += 1
# print("Acc:" + str(num_correct/total))

def formatPercentageToLR(yvector):
	y = yvector - 0.5
	y  = (np.sign(y) + 1)/2.0
	return y



def testLinearReg(trainX,trainY,testX,testY):
	print('Linear Regression')
	model = sklearn.linear_model.LinearRegression(fit_intercept=True)
	model.fit(trainX,trainY)
	print("Train R^2:" + str(model.score(trainX, trainY)))
	print("Test R^2:" + str(model.score(testX,testY)))	
	num_correct = 0
	total = 0
	for i in range(trainX.shape[0]):
	    x = np.array([trainX[i,:]]); y = trainY[i]; predictY = model.predict(x)[0]
	    if y < 0.5 and predictY < 0.5:
	    	num_correct += 1
	    if y > 0.5 and predictY > 0.5:
	    	num_correct += 1
	    total += 1
	print("Acc:" + str(num_correct/total))



def testLR(trainX,trainY, testX,testY):
	print('Logistic Regression')

	le = preprocessing.LabelEncoder()
	trainY = le.fit_transform(trainY)
	testY = le.transform(testY)


	model = sklearn.linear_model.LogisticRegression()
	model.fit(trainX,trainY)
	print("Test:",str(model.score(testX,testY)) )
	print('Train:',str(model.score(trainX,trainY)) )





#testLinearReg(trainX,trainY,testX,testY)


trainY = formatPercentageToLR(trainY)

testY = formatPercentageToLR(testY)



testLR(trainX,trainY,testX,testY)
