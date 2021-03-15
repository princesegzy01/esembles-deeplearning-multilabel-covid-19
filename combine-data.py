
import pandas as pd
import glob
from textblob import TextBlob
import sys
path = r'/home/michael/Documents/ML/multi-label-cnn-covid-19/data/processed' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv("0-427000.csv")
print("DOne")