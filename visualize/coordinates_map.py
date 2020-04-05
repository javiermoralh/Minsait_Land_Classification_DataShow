# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:33:04 2020

@author: javier.moral.hernan1
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

modelar = pd.read_csv(r'data/Modelar_UH2020.txt', sep='|')
estimar = pd.read_csv(r'data/Estimar_UH2020.txt', sep='|')
muestra_1 = modelar[0:200]

fig = plt.figure(figsize=(5, 4))
sns.scatterplot(x="X", y="Y", data=modelar)
fig

fig = plt.figure(figsize=(5, 4))
sns.scatterplot(x="X", y="Y", data=estimar)
fig


fig = plt.figure(figsize=(5, 4))
sns.scatterplot(x="X", y="Y", data=X_train)
fig

fig = plt.figure(figsize=(5, 4))
sns.scatterplot(x="X", y="Y", data=X_test)
fig