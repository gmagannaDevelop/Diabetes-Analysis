#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, NoReturn, Any, Callable, Union, Optional
import copy
import gc
import multiprocessing as mp

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 8)


# In[4]:


def time_indexed_df(df1: pd.core.frame.DataFrame, columname: str) -> pd.core.frame.DataFrame:
    """ 
        Cast into a time-indexed dataframe.
        df1 paramater should have a column containing datetime-like data,
        which contains entries of type pandas._libs.tslibs.timestamps.Timestamp
        or a string containing a compatible datetime (i.e. pd.to_datetime)
    """
    
    _tmp = df1.copy()
    
    pool = mp.Pool()
    _tmp[columname] = pool.map(pd.to_datetime, _tmp[columname])
    pool.close()
    pool.terminate()
    
    _tmp.index = _tmp[columname]
    _tmp.drop(columname, axis=1, inplace=True)
    _tmp = _tmp.sort_index()
    
    return _tmp
##


# In[22]:


x = pd.read_csv('data/CareLink-Export-03-ene-2020.csv')
x["DateTime"] =  x["Date"] + " " + x["Time"]
x.drop(["Date", "Time"], axis=1, inplace=True)


# In[23]:


y = time_indexed_df(x, 'DateTime')


# In[24]:


y = y.loc['2019-10-17':, :]


# In[25]:


y['Sensor Glucose (mg/dL)'].plot()


# In[26]:


sns.distplot(y['Sensor Glucose (mg/dL)'].dropna())


# In[27]:


hourly_mean = y['Sensor Glucose (mg/dL)'].groupby(y.index.hour).mean()
hourly_std  = y['Sensor Glucose (mg/dL)'].groupby(y.index.hour).std()


# In[28]:


hourly_mean.plot()


# In[13]:


y['Sensor Glucose (mg/dL)'].groupby(y.index.day).mean().plot()


# In[14]:


y['Sensor Glucose (mg/dL)'].groupby(y.index.day).std().plot()


# In[15]:


y['Sensor Glucose (mg/dL)'].groupby(y.index.hour).std().plot()


# In[ ]:





# In[16]:


hourly_std.apply(lambda x: x/hourly_std.shape[0])


# In[17]:


by_hour = y['Sensor Glucose (mg/dL)'].groupby(y.index.hour)
for i in by_hour:
    sns.scatterplot(i[1].dropna().index.hour, i[1].dropna(), label=f"{i[0]}")
    
plt.errorbar(
    hourly_mean.index, 
    hourly_mean.to_list(), 
    yerr=hourly_std.apply(lambda x: x).to_list(),
    c='green'
)

plt.axhline(160, c='red')
plt.axhline(80, c='red')


# In[18]:


z = y['Sensor Glucose (mg/dL)'].groupby(y.index.hour)


# In[19]:


for i in z:
    pass
    #sns.scatterplot(i[1].dropna().index.hour, i[1].dropna())
    #print(i[1].dropna())
    #plt.figure()
    #plt.title(f"Hour {i[0]}")
    #sns.distplot(i[1].dropna())


# In[20]:


y['Sensor Glucose (mg/dL)'].plot()


# In[21]:


y['Sensor Glucose (mg/dL)'].interpolate().plot()


# In[161]:


df = y['Sensor Glucose (mg/dL)']
df[np.bitwise_not(df.index.duplicated())]


# In[174]:


time_series = y['Sensor Glucose (mg/dL)']
time_series =  time_series[~df.index.duplicated()].interpolate()
print(time_series.dropna().head(), '\n\n', y['Sensor Glucose (mg/dL)'].dropna())
#time_series.plot()
#decomp = seasonal_decompose(time_series)


# In[148]:


help(seasonal_decompose)


# In[ ]:




