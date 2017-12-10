import pandas as pd 
import numpy as np 
import pdb
from os import listdir
from os.path import isfile, join



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

def get_pct_gop(county,y_data):
	# y_fname = "election_results.csv"

	# y = pd.read_csv(y_fname)

	y = y_data
	y["County"] = y["County"].str.lower()


	vals = y.loc[y['County']==county].per_gop.tolist()
	if len(vals) > 0:
		return vals[0]
	else:
		return("NA")


################################################################################################################################
# Collection Stage

def loadFilesFrom(folder):
	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	fnames = []
	for f in onlyfiles:
		if f[0] != '.' and ('.csv' in f):
			fnames.append(f)

	return loadFiles(folder,fnames)


def loadFiles(folder,fnames):
	data = None

	oldname = ''
	for fname in fnames:
		fname = folder + fname

		new = pd.read_csv(fname,encoding='mac_roman',header=0,index_col=0)
		# data.append(new)

		if type(data) == type(None):
			data = new
		else:
			data = data.join(new)

	return data

def loadAndExport(source,dest):
	data = loadFilesFrom(source)
	data.to_csv(dest)

def getYVals(fname):
	df = pd.read_csv(fname)
	return df


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

def removeXcols(X):
    df = X.copy(deep=True)

    for column in df:
	        # print(column)
            col = df[column]
	        # print('col:',col)
            val = col.iloc[5]
            if val == '(X)':
	            # print('True')
                df = df.drop(column, axis=1)
            elif val == '*****':
	            # print('True 2')
                df = df.drop(column, axis=1)
            elif 'MOE' in column:
	            # print('True 3')
                df = df.drop(column, axis=1)
            elif isinstance(col.iloc[0],str) and 'Margin' in col.iloc[0]:
	            # print('True 4')
                df = df.drop(column, axis=1)
            elif 'id' in column:
	            # print('True 5')
                df = df.drop(column, axis=1)
            elif 'Margin' in column:
                df = df.drop(column,axis=1)

    df.to_csv('./cleanedData.csv')
    return df


folder_name = './ogData/'
X = loadFilesFrom(folder_name)
X.to_csv('./combinedXdata.csv')
# print('total number columns:',len(list(X.columns.values)))

X = removeXcols(X)

y_name = './modified_elec_results.csv'
Y =  getYVals(y_name)
Y = Y.drop(Y.columns[0:1],axis=1)

df = addYcol(X,Y)
df.to_csv('./dataWithY.csv')

X12 = X.copy(deep = True)
Y12 = Y.copy(deep = True)


########################################################################################################
# Combining 2016 and 2012

fname2016 = './combinedData2016.csv'
mat16 = pd.read_csv(fname2016,encoding='mac_roman',header=0,index_col=0)

Y16 = mat16.iloc[:,-1]
mat16 = mat16.drop(mat16.columns[-1],axis=1)
X16 = mat16.copy(deep = True)

features16 = X16.columns
features12 = X12.columns

d = {}
for f12 in features12:
    for f16 in features16:
        if f16 == f12:
            d[f12] = True
    if not d.get(f12,False):
        d[f12] = False

 
