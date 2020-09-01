import pandas as pd
from preprocessing import preprocess_list
import numpy as np
#Canonical,Randomize

df = pd.read_csv('data/Data500k.csv')
dff = preprocess_list(df['Canonical'].tolist())
dff = dff.sample(frac=1).reset_index(drop=True)
nums = len(dff)
length = int(0.8*nums)
train_data = dff.iloc[0:length,:]
test_data = dff.iloc[(length+1):,:]

train_data.to_csv('data/train_data.csv')
test_data.to_csv('data/test_data.csv')
