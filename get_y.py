import pandas as pd

y = pd.read_csv("election_results.csv")
y["County"] = y["County"].str.lower()

def get_pct_gop(county):
    vals = y.loc[y['County']==county].per_gop.tolist()
    if len(vals) > 0:
        return vals[0]
    else:
        return("NA")
