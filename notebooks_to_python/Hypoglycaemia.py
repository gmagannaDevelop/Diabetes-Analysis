#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import toml
import multiprocessing as mp
from functools import reduce, partial

import pandas as pd
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

import copy
import gc

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.decomposition import PCA
from sklearn import preprocessing

from typing import List, Dict, NoReturn, Any, Callable, Union, Optional

from preproc import import_csv, new_hybrid_interpolator
from customobjs import objdict


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (15, 8)


# In[3]:


y = import_csv("interpolated/NG1988812H_Maganna_Gustavo_(27-05-20)_(9-06-20)_interpolated.csv")


# In[4]:


keyword = 'SUSPEND BEFORE LOW'
alarms  = []
for i in y.Alarm.dropna().unique().tolist():
    if keyword in i:
        alarms.append(i)
alarms


# In[9]:


dates = pd.unique(y.index.date)
n_total = len(dates)
print(f"Number of days in data : {len(dates)}")


# In[11]:


n_month = 30
n_latest = 4
#month = data.loc[dates[len(dates) - n_month]:dates[-1], :] if n_month < n_total else None
latest = y.loc[dates[len(dates)- n_latest]:dates[-1], :] if n_latest < n_total else None
lday = y.loc[dates[len(dates)- 1]:dates[-1], :] if n_latest < n_total else None


# In[12]:


latest[ latest.Alarm == 'SUSPEND BEFORE LOW ALARM, QUIET' ].hour.hist()


# In[16]:


latest.loc[str(dates[-2]), "Basal Rate (U/h)"] .plot()


# In[22]:


latest[latest.hour==22]["Basal Rate (U/h)"].describe()


# In[13]:


latest.columns


# In[ ]:




