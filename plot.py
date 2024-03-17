from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")
df.tail()

plt.plot(df['epoch'], df['mae'])
plt.show()
