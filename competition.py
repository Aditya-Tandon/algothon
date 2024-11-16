#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:51:58 2024

@author: mike
"""

import cryptpandas
import pandas as pd
import numpy as np
data_list = []
data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3547.crypt", password='oUFtGMsMEEyPCCP6'))
data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3611.crypt", password='GMJVDf4WWzsV1hfL'))
data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3675.crypt", password='PSI9bPh4aM3iQMuE'))
data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3739.crypt", password='1vA9LaAZDTEKPePs'))
data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3803.crypt", password='0n74wuaJ2wm8A4qC'))
data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3867.crypt", password='mXTi0PZ5oL731Zqx'))

data = pd.concat(data_list)
data = data[~data.isin([-np.inf, np.inf]).any(axis=1)]
data.min()
#%%
def get_positions(pos_dict):
    pos = pd.Series(pos_dict)
    pos = pos.replace([np.inf, -np.inf], np.nan)
    pos = pos.dropna()
    pos = pos / pos.abs().sum()
    pos = pos.clip(-0.1,0.1)
    if pos.abs().max() / pos.abs().sum() > 0.1:
        raise ValueError(f"Portfolio too concentrated {pos.abs().max()=} / {pos.abs().sum()=}")
    return pos

def get_submission_dict(
    pos_dict,
    your_team_name: str = "Team Physics",
    your_team_passcode: str = "Hamiltonian",
):
    
    return {
        **get_positions(pos_dict).to_dict(),
        **{
            "team_name": your_team_name,
            "passcode": your_team_passcode,
        },
    }
#%%
import matplotlib.pyplot as plt
# plt.figure()
for c in data.columns:
    # print(c)
    plt.figure()
    plt.plot(data[c].values,'.')
#%%
mu = []
for c in data.columns:
    mu.append(data[c].mean())
#%%
mu_total = sum(mu)
percentage = [m/mu_total for m in mu]
for i, p in enumerate(percentage):
    # print(p)
    if p > 0.1:
        percentage[i] = 0.1
    elif p< -0.1:
        percentage[i]= -0.1
print(percentage)
#%%
leftover_money = 1 - sum(percentage)

while leftover_money > 0:
    maxp= 0 
    maxi = None
    for i, p in enumerate(percentage):
        if p < 0.1 and p > maxp:
            maxp = p
            maxi = i
            # print('hey')
            # print(maxi)
            
    diff = 0.1 - maxp
    # print(maxp)
    if diff >= leftover_money:
        percentage[maxi] += leftover_money
        leftover_money = 0
    else:
        # print('test')
        # print(i)
        percentage[maxi] = 0.1
        leftover_money -=diff
        # print(leftover_money)
print(percentage)
#%%
d = {}
for i, c in enumerate(data.columns):
    d[c] = percentage[i]
#%%

#%%
d['team_name'] = 'Team Physics'
d['passcode'] = 'Hamiltonian'
#%%
history = []
#%%
"""
from dataset 3803 12:16 Nov 16 24
"""
# history = []
history.append(d)
# d
#%%
