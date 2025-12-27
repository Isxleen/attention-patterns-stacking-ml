import pandas as pd

flanker = pd.read_excel("data/raw/Raw_Flanker.xlsx")

flanker = flanker[flanker["latency"].between(200, 3000)] 
flanker = flanker[flanker["correct"].isin([0, 1])]

features_flanker = flanker.groupby("subjectid").agg({
    "latency": ["mean", "std"],
    "correct": "mean"
})

print(features_flanker.head())
print(features_flanker.columns)
print(features_flanker.info())
