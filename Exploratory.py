#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp
from functools import reduce

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

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
plt.style.use('ggplot')


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

def dist_plot(series: pd.core.series.Series, dropna: bool = True) -> NoReturn:
    """
        Given a pandas Series, generate a descriptive visualisation 
        with a boxplot and a histogram with a kde.
        By default, this function drops `nan` values. If you desire to
        handle them differently, you should do so beforehand and/or
        specify dropna=False.
    """
    
    if dropna:
        series = series.dropna()
    
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.25, .75)})
    sns.boxplot(series, ax=ax_box)
    sns.stripplot(series, color="orange", jitter=0.2, size=2.5, ax=ax_box)
    sns.distplot(series, ax=ax_hist, kde=True)
    ax_box.set(xlabel='')
##


# In[5]:


hba1c = lambda x: (x + 105) / 36.5


# In[48]:


get_duplicate_idx = lambda w: w[w.index.duplicated(keep=False)].index


# In[17]:


x = pd.read_csv('data/CareLink-Export-16-mar-2020.csv')
x["DateTime"] =  x["Date"] + " " + x["Time"]
x.drop(["Date", "Time"], axis=1, inplace=True)


# In[20]:


y = time_indexed_df(x, 'DateTime')
y.drop("Index", axis=1, inplace=True)


# In[21]:


duplicate_idx = y[y.index.duplicated(keep=False)].index
#duplicate_idx[1]


# In[57]:


#y.loc[duplicate_idx]
help(y.mask)


# In[59]:


z = y.mask( y == np.nan ).groupby(level=0).first()


# In[60]:


duplicate_idx2 = get_duplicate_idx(z)
duplicate_idx2


# In[61]:


y.loc[duplicate_idx[10]].T


# In[62]:


z.loc[duplicate_idx[10]].T


# In[ ]:


help(reduce)


# In[84]:


for i in duplicate_idx:
    print(y.loc[i])
    #reduce(lambda x, y: x.combine(y, lambda p, q: p if np.isnan(q) else q), y.loc[i])


# In[68]:


np.isnan(np.nan)


# In[ ]:


np.isnan


# In[53]:


y['hour'] = y.index.hour


# In[16]:


# Deltas within valuable intervals : 
for i in [10, 20, 30]: 
    y[f'd{i}'] = y['Sensor Glucose (mg/dL)'].diff(i)


# In[26]:


#help(pd.Series.diff)
len(y.groupby(y.index.day)["Sensor Glucose (mg/dL)"])


# In[28]:


plt.close("all")


# In[32]:


y[y.duplicated()]


# In[36]:


y.shape


# In[41]:


duplicates = y[y.index.duplicated(keep=False)].index


# In[46]:


help(y.combine)


# In[ ]:


lambda x:


# In[73]:


np.isnan(y.loc[duplicates[10], :]["New Device Time"][0])


# In[39]:


#y.groupby(level=0).filter(lambda x: len(x) > 1)


# In[18]:


idx = y['Sensor Glucose (mg/dL)'].dropna().index
y.loc[idx, ['Sensor Glucose (mg/dL)', 'd10', 'd20'] ].head(25)


# In[59]:


whole = y.copy()


# In[60]:


whole['ISIG Value'].dropna().count(), whole['Sensor Glucose (mg/dL)'].dropna().count()


# We can perform regression ! 

# In[11]:


bg_idx = whole['BG Reading (mg/dL)'].dropna().index
whole.loc[
    bg_idx - dt.timedelta(minutes=10) : bg_idx + dt.timedelta(minutes=10)
    , 'Sensor Glucose (mg/dL)'
]


# In[61]:


hba1c(whole['Sensor Glucose (mg/dL)'].dropna().mean())


# In[62]:


y = y.loc['2020-03-01':, :]


# In[63]:


hba1c(y['Sensor Glucose (mg/dL)'].dropna().mean())


# In[ ]:





# In[64]:


y['Sensor Glucose (mg/dL)'].plot()


# In[65]:


dist_plot(y['Sensor Glucose (mg/dL)'])


# In[ ]:





# In[66]:


y.columns


# In[67]:


keyword = 'SUSPEND BEFORE LOW'
alarms  = []
for i in y.Alarm.dropna().unique().tolist():
    if keyword in i:
        alarms.append(i)
alarms


# In[68]:


y[ y.Alarm == 'SUSPEND BEFORE LOW ALARM, QUIET' ].hour.hist()


# In[69]:


#meal_id = y['BWZ Carb Input (grams)'].dropna().index
nonull_meals = y['BWZ Carb Input (grams)'].dropna()
nonull_meals = nonull_meals[ nonull_meals > 0 ]
meal_id = nonull_meals.index
print(len(meal_id))
meal_id[:5]


# In[70]:


nonull_corrections = y['BWZ Correction Estimate (U)'].dropna()
nonull_corrections = nonull_corrections[ nonull_corrections > 0 ]
corrections_id = nonull_corrections.index
print(len(corrections_id))
corrections_id[:5]


# In[73]:


bolus_id = corrections_id.union(meal_id)
print(len(bolus_id))


# In[86]:


basal = y.copy()
for uid in bolus_id:
    real = uid+dt.timedelta(hours=2, minutes=30)
    closest = y.index[y.index.searchsorted(real) - 1]  # Otherwise it goes out of bounds !
    basal.loc[uid:closest, 'Sensor Glucose (mg/dL)'] = np.nan


# In[102]:





# In[90]:


y.loc['2020-03-15', 'Sensor Glucose (mg/dL)'].plot()
basal.loc['2020-03-15', 'Sensor Glucose (mg/dL)'].plot()


# In[92]:


basal.groupby(basal.index.hour)['Sensor Glucose (mg/dL)'].mean().plot()


# In[ ]:





# In[93]:


figs = [basal.groupby(basal.index.hour)[f'd{i}'].mean().plot(label=f"{i} min") for i in [10, 20, 30]]
figs[-1].legend()


# In[137]:


#y.loc['2020-03-14', 'Sensor Glucose (mg/dL)'].interpolate().plot()
#y.loc['2020-03-14', 'Sensor Glucose (mg/dL)'].interpolate(method='akima').plot()
y.loc['2020-03-14', 'Sensor Glucose (mg/dL)'].plot()


# In[ ]:





# In[90]:


abs(dt.timedelta(hours=1) - dt.timedelta(hours=2))


# In[26]:


y.loc[meal_id, 'BWZ Carb Ratio (g/U)'].dropna().index ==  meal_id


# In[27]:


dt10 = dt.timedelta(minutes=10)
dtpost_low = dt.timedelta(hours=1, minutes=40)
dtpost_high = dt.timedelta(hours=2, minutes=20)


# In[28]:


meal_descriptive = pd.core.frame.DataFrame({
    'hour': meal_id.hour, 
    'pre prandial': [ 
        y.loc[ meal - dt10 : meal + dt10,  'Sensor Glucose (mg/dL)' ].dropna().mean()
        for meal in meal_id
    ],
    'post mean': [
        y.loc[ meal + dtpost_low : meal + dtpost_high, 'Sensor Glucose (mg/dL)'].dropna().mean() 
        for meal in meal_id
    ],
    'post std': [
        y.loc[ meal + dtpost_low : meal + dtpost_high, 'Sensor Glucose (mg/dL)'].dropna().std() 
        for meal in meal_id
    ], 
    'post min': [
        y.loc[ meal + dtpost_low : meal + dtpost_high, 'Sensor Glucose (mg/dL)'].dropna().min() 
        for meal in meal_id
    ],
    'post max': [
        y.loc[ meal + dtpost_low : meal + dtpost_high, 'Sensor Glucose (mg/dL)'].dropna().max() 
        for meal in meal_id
    ],
}, index=meal_id)

meal_descriptive['delta'] = meal_descriptive['post mean'] - meal_descriptive['pre prandial'] 
meal_descriptive['ratio'] = y.loc[meal_id, 'BWZ Carb Ratio (g/U)'].dropna()


# In[29]:


meal_descriptive.loc[  meal_descriptive.hour < 6, 'meal' ] = 'night'
meal_descriptive.loc[ (meal_descriptive.hour >= 6)  & (meal_descriptive.hour < 9), 'meal'  ] = 'breakfast'
meal_descriptive.loc[ (meal_descriptive.hour >= 9)  & (meal_descriptive.hour < 12), 'meal' ] = 'lunch'
meal_descriptive.loc[ (meal_descriptive.hour >= 12) & (meal_descriptive.hour < 19), 'meal' ] = 'afternoon'
meal_descriptive.loc[ (meal_descriptive.hour >= 19) & (meal_descriptive.hour < 24), 'meal' ] = 'dinner'


# In[30]:


meal_descriptive.head()


# In[31]:


meal_descriptive[ meal_descriptive.meal ==  'lunch' ].delta.hist(rwidth=0.8)


# In[32]:


print(meal_descriptive[ meal_descriptive.meal == 'dinner'].describe())


# In[33]:


meal_descriptive.head()


# In[34]:


column    = 'delta'

print(column, '\n')
for i in set(meal_descriptive.meal):
    _tmp =  meal_descriptive[ meal_descriptive.meal == i ].describe().T
    print(i, '\n', f"Mean: {int(_tmp['mean'][column])}, Std: {int(_tmp['std'][column])}", '\n\n')


# In[35]:


postp = [
    y.loc[ 
        meal + dt.timedelta(hours=1, minutes=30) : meal + dt.timedelta(hours=3), 
        ['Sensor Glucose (mg/dL)', 'hour', *[f'd{i}' for i in range(1, 11)]] 
    ].dropna()
    for meal in meal_id
]

postp = pd.concat(postp)
postp.rename({
    'Sensor Glucose (mg/dL)': 'post points', 'b': 'Y'
}, axis='columns', inplace=True)

postp.head()


# In[36]:


postp.hour.hist()


# In[37]:


postp.groupby(postp.index.hour).mean().plot(
    x='hour', y='post points', kind='scatter', grid=True, xticks=list(range(24))
)
plt.axhline(150, color='blue')
plt.axhline(200, color='red')
plt.axhline(70, color='yellow')
postp.groupby(postp.index.hour).mean().plot(x='hour', y='post points', color='red')


# In[ ]:





# In[38]:


y.groupby('hour')['Sensor Glucose (mg/dL)'].mean()


# In[112]:


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




