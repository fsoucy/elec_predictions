import pandas as pd
import pdb

names = ['./data/income.csv','./data/population.csv','./data/race.csv','./data/education.csv']
for name in names:
    #import files into python
    
    df = pd.read_csv(name,encoding='mac_roman')
    df2 = df.set_index('Geography')
    # print('Columns:',list(df2.columns.values))
    # print('rows:',df2.index)
    # df = df2

    for column in df:
        val = df.at[5,column]
        if val == '(X)':
            # print('True')
            df = df.drop(column, 1)
        elif val == '*****':
            # print('True 2')
            df = df.drop(column, 1)
        elif 'MOE' in column:
            # print('True 3')
            df = df.drop(column, 1)
        elif isinstance(df.at[0,column],str) and 'Margin' in df.at[0,column]:
            # print('True 4')
            df = df.drop(column, 1)
        elif 'id' in column:
            # print('True 5')
            df = df.drop(column, 1)
    #remove first row
    # df.columns = df.iloc[0]
    # df = df.iloc[1:]
    df = df.set_index('Geography')
    print('Num Cols:',len(list(df.columns.values)))
    df.to_csv("./cleanedData" + name[6:-4]+'_removed.csv')
print('done')


"""
f = open(name, encoding='mac_roman')
    x = f.readlines(); f.close();
    lines = map(lambda l: l[:-1].split(","), x); lines = list(lines);
    print(len(lines)); print(len(lines[0]))
    for j in range(len(lines[0])):
        pdb.set_trace()
        val = lines[5][j]
        col = lines[0][j]
        if val == '(X)':
            lines = map(lambda l: l[:j] + l[j+1:], lines); lines = list(lines);
        elif val == '*****':
            lines = map(lambda l: l[:j] + l[j+1:], lines); lines = list(lines);
        elif 'MOE' in col:
            lines = map(lambda l: l[:j] + l[j+1:], lines); lines = list(lines);
        elif isinstance(col, str) and "Margin" in col:
            lines = map(lambda l: l[:j] + l[j+1:], lines); lines = list(lines);
        elif isinstance(col, str) and "id" in col:
            lines = map(lambda l: l[:j] + l[j+1:], lines); lines = list(lines);
    lines = map(lambda l: ''.join(x), lines); lines = list(lines);
    lines = map(lambda l: l + "\n", lines); lines = list(lines);
    output = '',join(lines)
    f = open("rem_data/" + name[5:-4]+"_removed.csv")
    f.write(output)
    f.close()
"""