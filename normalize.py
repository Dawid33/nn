from sklearn import preprocessing
import pandas as pd

df = pd.read_csv("reduced.csv")

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df2 = pd.DataFrame(x_scaled)
df2.to_csv("scaled.csv", header=df.columns, index=False)
