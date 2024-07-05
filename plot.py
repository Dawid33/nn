from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])
df.tail()

plt.plot(df['epoch'], df['mae'])
plt.show()
