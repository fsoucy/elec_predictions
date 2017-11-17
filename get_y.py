import pandas as pd



def get_pct_gop(county):
	y_fname = "election_results.csv"

	y = pd.read_csv(y_fname)
	y["County"] = y["County"].str.lower()


	vals = y.loc[y['County']==county].per_gop.tolist()
	if len(vals) > 0:
		return vals[0]
	else:
		return("NA")
