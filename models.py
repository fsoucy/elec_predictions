import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
import pdb
import get_y 
import sklearn.linear_model
from sklearn import preprocessing
from sklearn.linear_model import Lasso,Ridge, LogisticRegression
from sklearn.svm import SVC, LinearSVC
import combineData as cd 
import copy
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



folder = './cleanedData/'

X = cd.loadFilesFrom(folder)
#X = pd.read_csv('./data/education.csv',encoding='mac_roman')
#X = X.set_index("Geography")
cols = X.columns
colWorth = {}
for col in cols:
    colWorth[col] = 0.0
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
        if np.isnan(mat[i][j]):
            mat[i][j] = 0.0
mat = np.array(mat)
np.random.shuffle(mat)
X = mat[:,:-1]
Y = mat[:,-1]

def getAccuracy(yTrue, yPred):
    total = 0
    correct = 0
    for i, y in enumerate(yPred):
        total += 1
        if (y < 0.5) and (yTrue[i] < 0.5):
            correct += 1
        if (y > 0.5) and (yTrue[i] > 0.5):
            correct += 1
    return correct/total

def binarize(someY):
    otherY = someY.copy()
    otherY[otherY <= 0.5] = 0
    otherY[otherY > 0.5] = 1
    return otherY.astype('int')

trainX = X[:720,:]; trainY = Y[:720];
valX = X[600:720, :]; valY = Y[600:720];
testX = X[720:, :]; testY = Y[720:];

trainY, valY, testY = binarize(trainY), binarize(valY), binarize(testY)

Cs = [0.0000001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0]

model = XGBClassifier()
model.fit(trainX, trainY)
trainPred = model.predict(trainX)
testPred = model.predict(testX)
trainPred = [round(value) for value in trainPred]
testPred = [round(value) for value in testPred]
trainAccuracy = getAccuracy(trainY, trainPred)
testAccuracy = getAccuracy(testY, testPred)

print("Train accuracy: " + str(trainAccuracy))
print("Test accuracy: " + str(testAccuracy))

pdb.set_trace()


maxResult = 0.0; bestC = 0.0
for c in Cs:
    print(c)
    model = LinearSVC(C=c)
    model.fit(trainX, trainY)
    print("trained")
    result = model.score(valX, valY)
    print("scored")
    if result > maxResult:
        maxResult = result
        bestC = c

model = LinearSVC(C=bestC)
model.fit(trainX, trainY)
trainAcc = model.score(trainX, trainY)
valAcc = model.score(valX, valY)
testAcc = model.score(testX, testY)

print("Best C: " + str(bestC))
print("Train Accuracy: " + str(trainAcc))
print("Validation Accuracy: " + str(valAcc))
print("Test Accuracy: " + str(testAcc))

"""
binarizedTrainY = trainY.copy(); binarizedTestY = testY.copy();
binarizedTrainY[binarizedTrainY < 0.5] = 0.0
binarizedTrainY[binarizedTrainY > 0.5] = 1.0
binarizedTestY[binarizedTestY < 0.5] = 0.0
binarizedTestY[binarizedTestY > 0.5] = 1.0

model = XGBClassifier()
model.fit(trainX, binarizedTrainY)
trainPred = model.predict(trainX)
testPred = model.predict(testX)
trainPred = [round(value) for value in trainPred]
testPred = [round(value) for value in testPred]
trainAccuracy = getAccuracy(binarizedTrainY, trainPred)
testAccuracy = getAccuracy(binarizedTestY, testPred)
print(trainAccuracy)
print(testAccuracy)
output_file = "decisionTreeFeaturesCleaned.txt"
features = model.feature_importances_
feature_importances = []
output = ""
for i, feature in enumerate(cols):
    feature_importances.append((feature, features[i]))
feature_importances.sort(key=lambda x: x[1], reverse=True)
for feature, weight in feature_importances:
    output += feature + ": " + str(weight) + "\n"
f = open(output_file, "a")
f.write(output)
f.close()

model = Lasso(alpha = 0.0005, fit_intercept=True)
model.fit(trainX, trainY)
output = ""
output_file = "lassoFeaturesCleaned.txt"
features = model.coef_
feature_importances = []
for i, feature in enumerate(cols):
    feature_importances.append((feature, np.abs(features[i])))
feature_importances.sort(key=lambda x: x[1], reverse=True)
for feature, weight in feature_importances:
    output += feature + ": " + str(weight) + "\n"
f = open(output_file, "a")
f.write(output)
f.close()
pdb.set_trace()
"""


lams = [0.0000001] + [10.0, 25.0] + [50.0, 100.0, 200.0, 250.0, 300.0, 350.0, 400.0]
#lams = [0.0000001, 0.0001, 0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.05, 0.08, 0.1, 0.3, 0.5]
scores = []
for lam in lams:
    model = Ridge(alpha=lam, fit_intercept=True)
    model.fit(trainX, trainY)
    score = model.score(testX, testY)
    scores.append(score)

plt.plot(lams, scores, 'ro')
plt.xlabel("Lambda Regularization Coefficient")
plt.ylabel("R Squared Metric")
plt.title("Ridge regression - Uncleaned Data")
plt.savefig("ridge_uncleaned.png")
plt.show()
print(scores)
pdb.set_trace()
"""
X = X.T
np.random.shuffle(X)
X = X.T
"""

#if (X.shape[1] > 93):
#    X = X[:,:93]


#def determineEpsLam(X, Y, start, batchSize):
#    pass

def makeX(X, start, batchSize):
    currentX = X[:, start:start+batchSize]
    return currentX

def lassoDict(currentX, currentY, eps, lam, currentColumns, colWorth):
    irrelevant = []
    model = Lasso(alpha=lam, fit_intercept=True)
    model.fit(currentX, currentY)
    params = model.get_params()
    print(model.coef_.sum())
    for i in range(model.coef_.shape[0]):
        colWorth[currentColumns[i]] += np.abs(model.coef_[i])
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
eps = 0.01
lam = 0.008
start = 0
batchSize = 20
stride = 2
threshold = 0.1

# makeX, makeY
while (start+batchSize) < len(cols):
    currentX = makeX(X, start, batchSize)
    currentColumns = cols[start: start+batchSize]
    lasso_result = lassoDict(currentX, Y, eps, lam, currentColumns, colWorth)
    updateIrrelevant(lasso_result, irrelevantColumns)
    start += stride

doc_name = "cleanedWithLasso.csv"

originalData = outputNewFeatures(doc_name, originalData, irrelevantColumns, threshold, "Geography")

colWorthTups = []
for key in colWorth:
    colWorthTups.append((key, colWorth[key]))

colWorthTups.sort(key=lambda x: x[1], reverse=True)
output_file = "data_params_sorted_by_relevance.csv"
output = ""
for tup in colWorthTups:
    output += tup[0] + ": " + str(tup[1]) + "\n"
f = open(output_file, "a")
f.write(output)
f.close()

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
"""