# -*- coding: utf-8 -*-
# author: zxc time: 2020/4/24 15:25
import pandas as pd
import numpy as np
from preprocessing import randomize_smile

data = pd.read_csv('data/chem.txt',sep='\t',index_col=0)
print(data.head(3))
print(data.columns)
data.drop(columns=['a_heavy','a_acc', 'a_don', 'KierFlex'],inplace = True)
print(data.columns)
print(len(data))
print('开始处理')
"""
colums = ['ChEMBL ID', 'CAN_SMILES', 'Weight', 'logP(o/w)', 'a_heavy', 'apol',
       'a_ICM', 'balabanJ', 'h_pavgQ', 'logS', 'mr', 'vdw_vol', 'zagreb',
       'a_acc', 'a_don', 'KierFlex']
"""
for column in ['Weight', 'logP(o/w)', 'apol','a_ICM',
               'balabanJ', 'h_pavgQ', 'logS', 'mr', 'vdw_vol', 'zagreb' ]:
    data[column] = (data[column]-data[column].mean())/(data[column].max()-data[column].min())*2
random1 = data.copy()
random1['canonical_smiles'] = random1['CAN_SMILES']
random1.drop(columns=['CAN_SMILES'],inplace=True)
random1['random_smiles'] = random1['canonical_smiles']

randoms = random1.iloc[:int(0.94*(len(random1))),:]
randoms_test = random1.iloc[int(0.94*(len(random1))):,:]
randoms = [random1]*10
randoms_test = [randoms_test]*2
randoms = pd.concat(randoms,axis=0)
randoms_test = pd.concat(randoms_test,axis=0)
randoms['random_smiles'] = randoms['canonical_smiles'].map(randomize_smile)
randoms_test['random_smiles'] = randoms_test['canonical_smiles'].map(randomize_smile)

randoms.dropna(inplace=True)
randoms_test.dropna(inplace=True)

print(len(randoms['random_smiles'].unique()))
print(len(randoms_test['random_smiles'].unique()))
randoms.to_csv('data/chem_train.csv')
randoms_test.to_csv('data/chem_test.csv')
