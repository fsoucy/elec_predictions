import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb
import get_y



"""
Process for loading data:
1. Clean up csv files
2. combine files into test, val, and train matrices
3. rewrite as pickled data?
4. then do training

"""


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'District of Columbia':'DC',
    'Wyoming': 'WY',
}

def loadFilesFrom(folder):
	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	fnames = []
	for f in onlyfiles:
		if f[0] != '.' and ('.csv' in f):
			fnames.append(f)
	return loadFiles(folder,fnames)


def loadFiles(folder,fnames):
	data = pd.DataFrame([0])

	oldname = ''
	for fname in fnames:
		fname = folder + fname

		new = pd.read_csv(fname,encoding='mac_roman',header=0,index_col=0)
		# data.append(new)

		if len(data.index) == 1:
			data = new
		else:
			data = data.join(new,lsuffix='_data_'+oldname,rsuffix='_data_'+fname)

		oldname = fname
	return data

def loadAndExport(source,dest):
	data = loadFilesFrom(source)
	data.to_csv(dest)

def getYVals(fname):
	df = pd.read_csv(fname)
	return df

# fname = './election_results.csv'
# df = getYVals(fname)
# print(df)

def addYcol(X,y):
    X0 = X.copy()
    X0['y'] = None
    for index,row in X.iterrows():
        name = index
        county,state = name.split(', ')

        lis = county.split(' ')
        lis.pop()
        county = ' '.join(lis)

        county = county.replace('–','_')
        abbr = us_state_abbrev[state]


        vals = y.loc[y['County']==county].loc[y['State']==abbr].per_gop.tolist()

        if state == 'Louisiana':
            vals = y.loc[y['County']==(county + ' Parish')].loc[y['State']==abbr].per_gop.tolist()
        if state == 'Virginia' and len(vals) != 1:
            vals = y.loc[y['County']==(county + ' city')].loc[y['State']==abbr].per_gop.tolist()


        if len(vals) == 1:
            X0.set_value(index,'y',vals[0])
    return X0



X = loadFiles('./cleanedData/',['education_removed.csv'])
y = pd.read_csv('./election_results.csv')
X = addYcol(X,y)
X.to_csv('./2012/combinedData2016.csv')










# folder = './data/'
# fnames = ['population.csv']

# X = loadFilesFrom(folder)
# y = pd.read_csv('./election_results.csv')
# X = addYcol(X,y)

# mat = X.as_matrix()
# mat = list(mat)
# for i in range(len(mat)):
#     for j in range(len(mat[i])):
#         val = mat[i][j]
#         try:
#             mat[i][j] = float(val)
#         except:
#             mat[i][j] = 0.0
# mat = np.array(mat)
# np.random.shuffle(mat)
# X = mat[:,:-1]
# X = X.T
# np.random.shuffle(X)
# X = X.T
# X = X[:,:250]
# Y = mat[:,-1]
# trainX = X[:600,:]; trainY = Y[:600];
# testX = X[600:, :]; testY = Y[600:];


# import sklearn.linear_model
# model = sklearn.linear_model.LinearRegression(fit_intercept=True)
# model.fit(trainX,trainY)
# print("Train R^2:" + str(model.score(trainX, trainY)))
# print("Test R^2:" + str(model.score(testX,testY)))

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

# pdb.set_trace()

# X.insert(0,'names',X.index)
# mat = X.as_matrix()
# for row in mat:
	# print(str(row).encode('ascii','ignore'))

# fname = './test.csv'
# X.to_csv(fname)



# newnames = []

# for index,row in X.iterrows():
# 	name = index
# 	county,state = name.split(', ')column
# 	lis = county.split(' ')
# 	lis.pop()
# 	county = ' '.join(lis)
# 	county = county.replace('–','_')
# 	abbr = us_state_abbrev[state]

# 	vals = y.loc[y['County']==county].loc[y['State']==abbr].per_gop.tolist()
# 	if len(vals) == 1:
# 		X.set_value(index,'y',vals[0])
# 	newnames.append((county,abbr))

# print(X['y'])


# names = X.index
# newnames =[]
# for name in names:
# 	county,state = name.split(', ')
# 	lis = county.split(' ')
# 	lis.pop()
# 	county = ' '.join(lis)
# 	county = county.replace('–','_')
# 	abbr = us_state_abbrev[state]
# 	newnames.append((county,abbr))

# lis = []
# for county,abbr in newnames:
# 	print('#########################')
# 	print('County:',county)
# 	vals = y.loc[y['County']==county].loc[y['State']==abbr].per_gop.tolist()
# 	lis.append(vals)
# print(lis)
