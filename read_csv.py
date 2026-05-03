import pandas as pd

df = pd.read_csv("data/loop_dataset.csv")
print(df.shape)
print(df.columns)
print(df.head())