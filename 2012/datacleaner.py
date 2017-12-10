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


folder_name = './eduData/'
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

def getSimilar(feat12,feat16):
    """
    feat12 and feat16 are feature strings. 
    """

    redundant = ['all','(dollars)','(SSI)','or','and','-','some','is','for','whom','by','attainment','level','higher','imputed','allocated']

    f12 = feat12
    f16 = feat16

    while '(' in f12 or ')' in f12:
        if '(' in f12:
            ind = f12.index('(')
            f12 = f12[:ind] + f12[ind + 1:]
        if ')' in f12:
            ind = f12.index(')')
            f12 = f12[:ind] + f12[ind + 1:]

    while '(' in f16 or ')' in f16:
        if '(' in f16:
            ind = f16.index('(')
            f16 = f16[:ind] + f16[ind + 1:]
        if ')' in f16:
            ind = f16.index(')')
            f16 = f16[:ind] + f16[ind + 1:]


    list12 = f12.split('; ')
    list16 = f16.split('; ')

    temp = []
    for e in list12:
        temp.append(e.split(' '))
    temp = [item.lower() for sublist in temp for item in sublist]
    new = []
    for word in temp:
        if word not in redundant:
            if word == 'family' or word == 'families':
                new.append('households')
            elif word == 'household':
                new.append('households')
            else:
                new.append(word)

    list12 = new

    temp = []
    for e in list16:
        temp.append(e.split(' '))
    temp = [item.lower() for sublist in temp for item in sublist]
    new = []
    for word in temp:
        if word not in redundant:
            if word == 'family' or word == 'families':
                new.append('households')
            elif word == 'household':
                new.append('households')
            else:
                new.append(word)

    list16 = new

    check12in16 = False
    check16in12 = False
    count12 = 0
    count16 = 0

    diff = []

    for f12 in list12:
        if f12 in list16:
            count12 += 1
        else:
            diff.append(f12)

    for f16 in list16:
        if f16 in list12:
            count16 += 1
        else:
            diff.append(f16)

    if len(list12) == count12:
        check12in16 = True
    if len(list16) == count16:
        check16in12 = True

    if check16in12 and check12in16:
        check = True
        print('Check is True')
    else:
        print('check12in16:', check12in16)
        print('check16in12:',check16in12)
        check = False

    print('List 12:',list12)
    print('List 16:',list16)
    print('Diff:',diff)
    if len(diff) == 1:
        check = True
    return check

    


fname2016 = './combinedData2016.csv'
mat16 = pd.read_csv(fname2016,encoding='mac_roman',header=0,index_col=0)

Y16 = mat16.iloc[:,-1]
mat16 = mat16.drop(mat16.columns[-1],axis=1)
X16 = mat16.copy(deep = True)

features16 = list(X16.columns)
features12 = list(X12.columns)

# features16 = features16[:10]
# features12 = features12[:10]

print('features12:',len(features12))
print('features16:',len(features16))

d = {}
count = 0
for f12 in features12:
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    for f16 in features16:
        print('###########################################################')
        print('f12:',f12)
        print('f16:',f16)
        out  = getSimilar(f12,f16)
        if out == True and f16 not in d.values():
            d[f12] = f16
            count += 1
            break
    if not d.get(f12,False):
        d[f12] = False

for k,v in d.items():
    if v is not False:
        print('#############################################')
        print('key:',k)
        print('value:',v)
print('count:',count)
 
