import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
import pdb
import get_y 
import sklearn.linear_model
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import combineData as cd 
import copy




<<<<<<< HEAD
folder = './cleanedData/'
=======
folder = './data/'
>>>>>>> d95ea0f35ca30f37729c0d7e3013a87bc95465db

#X = cd.loadFilesFrom(folder)
X = pd.read_csv('./data/education.csv',encoding='mac_roman')
X = X.set_index("Geography")
cols = X.columns
originalData = copy.deepcopy(X)
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
#np.random.shuffle(mat)
X = mat[:,:-1]
Y = mat[:,-1]
#Y = Y * 10000.0

X = X.T
np.random.shuffle(X)
X = X.T

if (X.shape[1] > 93):
    X = X[:,:93]
"""

#def determineEpsLam(X, Y, start, batchSize):
#    pass

def makeX(X, start, batchSize):
    currentX = X[:, start:start+batchSize]
    return currentX

def lassoDict(currentX, currentY, eps, lam, currentColumns):
    irrelevant = []
    model = Lasso(alpha=lam, fit_intercept=True)
    model.fit(currentX, currentY)
    params = model.get_params()
    print(model.coef_.sum())
    for i in range(model.coef_.shape[0]):
        if np.abs(model.coef_[i]) < eps:
            irrelevant.append(currentColumns[i])
    return irrelevant

def updateIrrelevant(lasso_result, irrelevantColumns):
    for feature in lasso_result:
        if feature in irrelevantColumns:
            irrelevantColumns[feature] += 1
        else:
            irrelevantColumns[feature] = 1

def outputNewFeatures(doc_name, originalData, irrelevantColumns, threshold, ind):
    print("Originally had " + str(len(originalData.columns)) + " features.")
    for feature in irrelevantColumns:
        if irrelevantColumns[feature] > threshold and feature != ind:
            originalData = originalData.drop(feature, 1)
    print("Now have " + str(len(originalData.columns)) + " features.")
    originalData.to_csv(doc_name)
    return originalData

irrelevantColumns = {}

# eps, lam = determineEpsLam(X, Y, start, batchSize)
eps = 100.0
lam = 100.0
start = 0
batchSize = 5
stride = 1
threshold = 3

# makeX, makeY
while (start+batchSize) < len(cols):
    currentX = makeX(X, start, batchSize)
    currentColumns = cols[start: start+batchSize]
    lasso_result = lassoDict(currentX, Y, eps, lam, currentColumns)
    updateIrrelevant(lasso_result, irrelevantColumns)
    start += stride

doc_name = "edu_test.csv"

originalData = outputNewFeatures(doc_name, originalData, irrelevantColumns, threshold, "Geography")


mat = originalData.as_matrix()
mat = list(mat)
for i in range(len(mat)):
    for j in range(len(mat[i])):
        val = mat[i][j]
        try:
            mat[i][j] = float(val)
        except:
            mat[i][j] = 0.0
mat = np.array(mat)
X = mat
Y = Y / 10000.0
"""


print("Num features: " + str(X.shape[1]))
trainX = X[:600,:]; trainY = Y[:600];
testX = X[600:, :]; testY = Y[600:];

print(trainX.shape)


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
