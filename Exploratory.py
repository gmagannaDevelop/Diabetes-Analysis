#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


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


# In[169]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('ggplot')


# In[255]:


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
    """
    
    if dropna:
        series = series.dropna()
    
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.25, .75)})
    sns.boxplot(series, ax=ax_box)
    sns.stripplot(series, color="orange", jitter=0.2, size=2.5, ax=ax_box)
    sns.distplot(series, ax=ax_hist, kde=True)
    ax_box.set(xlabel='')
##


# In[192]:


x = pd.read_csv('data/CareLink-Export-03-ene-2020.csv')
x["DateTime"] =  x["Date"] + " " + x["Time"]
x.drop(["Date", "Time"], axis=1, inplace=True)


# In[193]:


y = time_indexed_df(x, 'DateTime')
y['hour'] = y.index.hour


# In[194]:


# Deltas within minutes
for i in range(1, 11):
    y[f'd{i}'] = y['Sensor Glucose (mg/dL)'].diff(i)


# In[195]:


y = y.loc['2019-12-25':, :]


# In[196]:


y['Sensor Glucose (mg/dL)'].plot()


# In[256]:


dist_plot(y['Sensor Glucose (mg/dL)'])


# In[405]:


y.columns


# In[332]:


keyword = 'SUSPEND BEFORE LOW'
alarms  = []
for i in y.Alarm.dropna().unique().tolist():
    if keyword in i:
        alarms.append(i)
alarms


# In[344]:


y[ y.Alarm == 'SUSPEND BEFORE LOW ALARM, QUIET' ].hour.hist()


# In[403]:


#meal_id = y['BWZ Carb Input (grams)'].dropna().index
nonull_meals = y['BWZ Carb Input (grams)'].dropna()
nonull_meals = nonull_meals[ nonull_meals > 0 ]
meal_id = nonull_meals.index
print(len(meal_id))


# In[410]:


y.loc[meal_id, 'BWZ Carb Ratio (g/U)'].dropna().index == meal_id


# In[386]:


dt10 = dt.timedelta(minutes=10)
dtpost_low = dt.timedelta(hours=1, minutes=45)
dtpost_high = dt.timedelta(hours=2, minutes=45)


# In[387]:


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


# In[388]:


meal_descriptive.loc[  meal_descriptive.hour < 6, 'meal' ] = 'night'
meal_descriptive.loc[ (meal_descriptive.hour >= 6)  & (meal_descriptive.hour < 9), 'meal'  ] = 'breakfast'
meal_descriptive.loc[ (meal_descriptive.hour >= 9)  & (meal_descriptive.hour < 12), 'meal' ] = 'lunch'
meal_descriptive.loc[ (meal_descriptive.hour >= 12) & (meal_descriptive.hour < 19), 'meal' ] = 'afternoon'
meal_descriptive.loc[ (meal_descriptive.hour >= 19) & (meal_descriptive.hour < 24), 'meal' ] = 'dinner'


# In[389]:


ratios = {
    'night': 12.0,
    'breakfast': 10.0, 
    'lunch': 12.0,
    'afternoon': 12.0,
    'dinner': 18.0
}


# In[390]:


for key, value in ratios.items():
    meal_descriptive.loc[meal_descriptive.meal == key, 'ratio'] = value 


# In[399]:


print(meal_descriptive[ meal_descriptive.meal == 'dinner'].describe())


# In[315]:


meal_descriptive.head()


# In[317]:


column    = 'delta'

print(column, '\n')
for i in set(meal_descriptive.meal):
    _tmp =  meal_descriptive[ meal_descriptive.meal == i ].describe().T
    print(i, '\n', f"Mean: {int(_tmp['mean'][column])}, Std: {int(_tmp['std'][column])}", '\n\n')


# In[268]:


postp = [
    y.loc[ 
        meal + dt.timedelta(hours=1, minutes=30) : meal + dt.timedelta(hours=3), 
        ['Sensor Glucose (mg/dL)', 'hour', *[f'd{i}' for i in range(1, 11)]] 
    ].dropna()
    for meal in meal_id
]

postp = pd.concat(postp)
postp.rename({'Sensor Glucose (mg/dL)': 'post points', 'b': 'Y'}, axis='columns', inplace=True)
postp.head()


# In[204]:


postp.hour.hist()


# In[177]:


postp.groupby(postp.index.hour).mean().plot(
    x='hour', y='Sensor Glucose (mg/dL)', kind='scatter', grid=True, xticks=list(range(24))
)
plt.axhline(150, color='blue')
plt.axhline(200, color='red')
plt.axhline(70, color='yellow')
postp.groupby(postp.index.hour).mean().plot(x='hour', y='Sensor Glucose (mg/dL)', color='red')


# In[107]:


y['Sensor Glucose (mg/dL)'].diff(10).groupby(y.index.hour).mean().plot()


# In[187]:





# In[190]:


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




