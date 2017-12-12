#perform pca to get best k features
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm

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
Y = np.round(Y)
temp2 = np.nan_to_num(Y)
#remove elements with no target
for i in range(len(temp2)):
    if temp2[i] == type('NaN') or np.isnan(temp[i,j]) :
        del X[i]
        del temp2[i]
Y = temp2.copy().astype(float)
print(Y)
X = scaler.fit_transform(X)
k = 10

#train test split
Xtrain,Xtest, ytrain, ytest = train_test_split(X,Y,test_size =0.2)

sumvar = []
nnacs = []
for k in range(1,120):
    print('k = ' + str(k))
    pca = PCA(n_components = k)
    pca.fit(Xtrain)
    var = pca.explained_variance_ratio_
    sums = sum(var)
    #record variance explained by first k features
    sumvar.append(sums)
    #get new X feature vectors
    Xnew = pca.fit_transform(Xtrain)
    Xtestnew = pca.transform(Xtest)
    #fit neural net to X, Y
    arch1: k
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0
    for i in range(20):
        clf1 = MLPClassifier(hidden_layer_sizes=(k),early_stopping=True)
        clf1.fit(Xnew,ytrain)
        #arch 2: k/2
        clf2 = MLPClassifier(hidden_layer_sizes=(max((1,int(k/2)))),early_stopping=True)
        clf2.fit(Xnew,ytrain)
        #arch3: k/2, k/2
        clf3 = MLPClassifier(hidden_layer_sizes=(max((1,int(k/2))),max((1,int(k/2)))),early_stopping=True)
        clf3.fit(Xnew,ytrain)
        clf4 = svm.SVC(kernel='rbf')
        clf4.fit(Xnew,ytrain)
        #evaluate accuracy, save to array
        acc1 += clf1.score(Xtestnew,ytest)
        acc2 += clf2.score(Xtestnew,ytest)
        acc3 += clf3.score(Xtestnew,ytest)
        acc4 += clf4.score(Xtestnew,ytest)
    accs = (acc1/20,acc2/20,acc3/20, acc4/20)
    nnacs.append(accs)
    #train gaussian kernel model to X, Y
    #evaluate accuracy, save to array


print (sum(var))
plt.plot(sumvar)
plt.title('Feature Reduction via PCA')
plt.xlabel('k')
plt.ylabel('percent variance explained by first k features')
plt.show()

plt.clf()
plt.title('Neural Network Accuracies ')
plt.xlabel('Number of PCA features')
plt.ylabel('Accuracy')
arch1 = [a[0] for a in nnacs]
arch2 = [a[1] for a in nnacs]
arch3 = [a[2] for a in nnacs]
arch4 = [a[3] for a in nnacs]
plt.plot(arch1,label='k',color='blue')
plt.plot(arch2,label='k/2',color='black')
plt.plot(arch3,label='k/2,k/2',color='red')
plt.plot(arch4, label ='gaussian SVM', color='yellow')
plt.legend()
plt.show()
