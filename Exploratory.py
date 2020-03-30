#!/usr/bin/env python
# coding: utf-8

# In[223]:


import os
import multiprocessing as mp
from functools import reduce, partial

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[3]:


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


# In[4]:


print(plt.style.available)
styles = plt.style.available


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (15, 8)


# In[147]:


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


def merge_on_duplicate_idx(
    df: pd.core.frame.DataFrame, 
    mask: Any = np.nan,
    verbose: bool = False
) -> pd.core.frame.DataFrame:
    """
    """
    
    y = df.copy()
    y = y.mask( y == mask ).groupby(level=0).first()
    
    if verbose:
        original_rows = df.shape[0]
        duplicate_idx = df[df.index.duplicated(keep=False)].index
        duplicate_rows = df.loc[duplicate_idx].shape[0]
        new_rows = y.shape[0]
        print(f" Total rows on source dataframe :\t{original_rows}")
        print(f" Duplicate indices :\t\t\t{duplicate_idx.shape[0]}")
        print(f" Total duplicate rows :\t\t\t{duplicate_rows}")
        print(f" Rows on pruned dataframe :\t\t{new_rows}")
    
    return y
##

def comparative_hba1c_plot(
    df: pd.core.frame.DataFrame,
    colum_name: str = "Sensor Glucose (mg/dL)",
    hba1c: Callable = lambda x: (x + 105) / 36.5,
    windows: Dict[str,int] = {
        "weekly": 7,
        "monthly": 30
    }
) -> NoReturn:
    """
        
    """
    df.groupby(df.index.dayofyear)[colum_name].mean().apply(hba1c).plot(**{"label":"daily"})
    for key, value in windows.items():
        df.groupby(df.index.dayofyear)[colum_name].mean().rolling(value).mean().apply(hba1c).plot(**{"label":key})
    mean_hba1c = hba1c(df[colum_name].mean()) 
    plt.axhline(mean_hba1c, **{"label": f"mean = {round(mean_hba1c,1)}", "c": "blue"})
    plt.legend()
##

def probability_estimate(
    data: pd.core.series.Series, 
    start: float, 
    end: float, 
    N: int = 250,
    percentage: bool = False,
    show_plots: bool = False
) -> float:
    """
        buggy.
    """
    
    # Plot the data using a normalized histogram
    dev = copy.deepcopy(data)
    dev = dev.dropna().apply(int)
    
    x = np.linspace(dev.min(), min(data), max(data))[:, np.newaxis]

    # Do kernel density estimation
    kd = KernelDensity(kernel='gaussian', bandwidth=0.85).fit(np.array(dev).reshape(-1, 1))

    # Plot the estimated densty
    kd_vals = np.exp(kd.score_samples(x))

    # Show the plots
    if show_plots:
        plt.plot(x, kd_vals)
        plt.hist(dev, 50, normed=True)
        plt.xlabel('Concentration mg/dl')
        plt.ylabel('Density')
        plt.title('Probability Density Esimation')
        plt.show()

    #probability = integrate(lambda x: np.exp(kd.score_samples(x.reshape(-1, 1))), start, end)[0]
    
    # Integration :
    step = (end - start) / (N - 1)  # Step size
    x = np.linspace(start, end, N)[:, np.newaxis]  # Generate values in the range
    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
    
    if percentage:
        probability *= 100
    
    return probability
##

def hybrid_interpolator(
    data: pd.core.series.Series,
    mean: float = None,
    limit: float = None,
    methods: List[str] = ['linear', 'spline'], 
    weights: List[float] = [0.65, 0.35],
    direction: str = 'forward',
    order: int = 2
) -> pd.core.series.Series:
    """
    Return a pandas.core.series.Series instance resulting of the weighted average
    of two interpolation methods.
    
    Model:
        φ = β1*method1 + β2*method2
        
    Default:
        β1, β2 = 0.6, 0.4
        method1, method2 = linear, spline
    
    Weights are meant to be numbers from the interval (0, 1)
    which add up to one, to keep the weighted sum consistent.
    
    limit_direction : {‘forward’, ‘backward’, ‘both’}, default ‘forward’
    If limit is specified, consecutive NaNs will be filled in this direction.
    
    If the predicted φ_i value is outside of the the interval
    ( (mean - limit), (mean + limit) )
    it will be replaced by the linear interpolation approximation.
    
    If not set, mean and limit will default to:
        mean = data.mean()
        limit = 2 * data.std()
    
    This function should have support for keyword arguments, but is yet to be implemented.
    """
    predictions: List[float] = [] 
    
    if not np.isclose(sum(weight for weight in weights), 1):
        raise Exception('Sum of weights must be equal to one!')
    
    for met in methods:
        if (met == 'spline') or (met == 'polynomial'):
            predictions.append(data.interpolate(method=met, order=order, limit_direction=direction))
        else:
            predictions.append(data.interpolate(method=met, limit_direction=direction))

    linear: pd.core.series.Series = predictions[0]
    spline: pd.core.series.Series = predictions[1]
    hybrid: pd.core.series.Series = weights[0]*predictions[0] + weights[1]*predictions[1]
    
    corrected: pd.core.series.Series = copy.deepcopy(hybrid) 
    
    if not mean:
        mean = data.mean()
    if not limit:
        limit = 2 * data.std()
    
    for idx, val in zip(hybrid[ np.isnan(data) ].index, hybrid[ np.isnan(data) ]):
        if (val > mean + limit) or (val < mean - limit):
            corrected[idx] = linear[idx]
    
    #df = copy.deepcopy(interpolated)
    #print(df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum())
    
    return corrected
##


# ### HBA1C%
# The gold standard for diabetes diagnosis and evaluation of its therapy. See the [Mayo clinic's page on HBA1C](https://www.mayocliniclabs.com/test-catalog/Clinical+and+Interpretive/82080)

# In[7]:


hba1c = lambda x: (x + 105) / 36.5


# In[8]:


#get_duplicate_idx = lambda w: w[w.index.duplicated(keep=False)].index


# In[338]:


get_csv_files = lambda loc: [os.path.join(loc, x) for x in os.listdir(loc) if x[-4:] == ".csv"] 


# In[339]:


files = get_csv_files("data/newest/")
files


# In[340]:


the_file = files[0]
x = pd.read_csv(the_file)
# x.columns


# Read the csv file, and inspect the columns. Our ```time_indexed_df``` function requires a single column containing "Datetime". For this purpose we calculate it by concatenating the **Date** and **Time** columns from the csv.

# In[341]:


x["DateTime"] =  x["Date"] + " " + x["Time"]
x.drop(["Date", "Time"], axis=1, inplace=True)


# In[342]:


x.head(3)


# Here we create a *date_time-indexed dataframe*. Afterwards we drop the original, useless, "Index" column. We then merge on duplicate indices as the Pump system logs a separate entry for each event, even when they occur simultaneously.

# In[343]:


y = time_indexed_df(x, 'DateTime')
y.drop("Index", axis=1, inplace=True)
# experimental  
y.index = y.index.map(lambda t: t.replace(second=0))
# end experimental
y = merge_on_duplicate_idx(y, verbose=True)
z = y.copy()


# In[344]:


# TMP !
z = y.loc["2020-03-12":"2020-03-27", :]


# If you would like to better understand the *merge* mechanism, uncomment the following code snippet, index the dataframe on the pertinent indices (```duplicate_idx``` variable).

# In[34]:


# Removing duplicates : 
"""
duplicate_idx = y[y.index.duplicated(keep=False)].index
print(duplicate_idx.shape)
y = y.mask( y == np.nan ).groupby(level=0).first()
"""


# In[38]:


# Remove unnecessary seconds resolution from datetime-index : 
#y.index = y.index.map(lambda t: t.replace(second=0))


# In[120]:


pd.infer_freq(y.index)
adv_methods = ['krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima']


# In[345]:


#help(hybrid_interpolator)


# In[418]:


clip = pd.read_clipboard(sep='\s\s+')
clip.index = clip.index.map(pd.to_datetime)


# In[426]:


clip.index.to_series().diff(1) > dt.timedelta(minutes=45)


# In[423]:


clip.index


# In[ ]:





# In[346]:


w = z.loc[:, ["Sensor Glucose (mg/dL)", "ISIG Value"]]#.resample("1T").asfreq()
#v = w.interpolate(method=adv_methods[-4], order=4)
v = hybrid_interpolator(
    w["Sensor Glucose (mg/dL)"],
    weights=[0.5, 0.5]
)


# In[415]:


w.loc['2020-03-16 16:31:00':'2020-03-16 19:30:00', ["Sensor Glucose (mg/dL)", "ISIG Value"]]


# In[347]:


start = "2020-03-16 9:35"
stop  = "2020-03-17 19:45"


# In[391]:


#help(pd.core.frame.DatetimeIndex.to_series)


# In[404]:


#dir(w.index)
indices_df = pd.DataFrame(w.index.to_series(name="Index"))
indices_df["Deltas"] = indices_df.Index.diff(1)
indices_df["Cutoff"] = indices_df.Deltas > dt.timedelta(minutes=45)
indices_df.Cutoff[indices_df.Cutoff == True].index.to_list()


# In[397]:


indices_df["Deltas"] = indices_df.Index.diff(1)


# In[349]:


greater_than_half = indices.diff(1) > dt.timedelta(minutes=45)


# In[ ]:





# In[350]:


jumps = greater_than_half[greater_than_half == True].index.to_list()
jumps


# In[351]:


#w.index
g = w.applymap(lambda x: int(x) if not np.isnan(x) else x).groupby("ISIG Value")
g


# In[379]:


g = w.groupby(level=0, by=jumps)


# In[380]:


for i, j in g:
    print(i)
    print(j.shape)


# In[365]:


g = w.index.groupby(jumps)


# In[370]:


keys = list(g.keys())


# In[375]:


g[keys[0]]


# In[376]:


#w.loc[g[], :]


# In[336]:


for i, j in g:
    print(i)


# In[298]:


for cut_date, split_df in w.groupby(jumps, axis=0):
    print(cut_date)


# In[257]:


#help(pd.DataFrame.groupby)


# In[246]:


help(pd.DataFrame.between_time)


# In[234]:


cutter


# In[430]:


#v.loc[start:stop].plot()
z.loc[start:stop, "Sensor Glucose (mg/dL)"].interpolate().plot(label='Linear iinterpolation')
z.loc[start:stop, "Sensor Glucose (mg/dL)"].plot(label='Original')
z.loc[star
plt.legend()


# In[115]:


z["Sensor Glucose (mg/dL)"].dropna().index == w["Sensor Glucose (mg/dL)"].dropna().index


# In[435]:


start2 = "2020-03-13"
stop2  = "2020-03-16"


# In[437]:


z.loc["2020-03-14":"2020-03-14", "Sensor Glucose (mg/dL)"].interpolate("linear").plot()
z.loc[start2:stop2, "Sensor Glucose (mg/dL)"].interpolate("cubic").plot()
z.loc[start2:stop2, "Sensor Glucose (mg/dL)"].plot()


# In[117]:


#help(y.resample)
#w = y.resample("1T").apply(lambda x: x)


# In[45]:


#dir(y.index)


# In[48]:


y.index[:10], w.index[:10]


# In[88]:


y.loc[meal_id, :].shape, w.loc[meal_id, :].shape


# In[65]:


# Useful having an hour column, for groupby opperations :
y['hour'] = y.index.hour
# Deltas within valuable intervals : 
for i in [10, 20, 30]: 
    y[f'd{i}'] = y['Sensor Glucose (mg/dL)'].diff(i)


# The previous code snippet generates *delta* columns. These are, however, suboptimal as logging seems to be inconsistent. This may be fixed via interpolation, but precaution is mandatory.

# The following attemps to *map* the hours and minutes from the datetime index to a [parametric circle](https://mathopenref.com/coordparamcircle.html). You might ask : **Why?**
# 
# The insulin pump is configured to adjust the hourly basal dose of insulin. In general, a healthy pancreas would constantly secrete insulin responding to subtle variations in glycaemia. When there is a carb intake, i.e. a the person eats something such as bread, fruit, etc. glycaemia rises and this augmentation frees insulin to the bloodstream.
# 
# Insulin sensibility varies throughout the day. The previously mentioned pump configuration has scheduled basal doses and **insulin-glycaemia-drop ratio, a.k.a insulin sensitivity** and **insulin-carbohidrate absorption ratio, alias carb ratio**.
# 
# To better represent this periodicity, I've decided to create this two new periodic variables as the sine and cosine of the hour and minute of the day. This enables the expression of the periodicity of physiological phenomena, i.e. today's midnight is closer to tomorrow's morning than it is to the same day's morning.

# In[66]:


T = 1439
min_res_t_series = pd.Series(y.hour*60 + y.index.minute)
y['x(t)'] = min_res_t_series.apply(lambda x: np.cos(2*np.pi*(x) / T))
y['y(t)'] = min_res_t_series.apply(lambda x: np.sin(2*np.pi*(x) / T))
# sns.scatterplot(x="x(t)", y="y(t)", data=y)


# In[67]:


#y.columns


# In[68]:


idx = y['Sensor Glucose (mg/dL)'].dropna().index
#y.loc[idx, ['Sensor Glucose (mg/dL)', 'd10', 'd20'] ].head(25)


# In[69]:


whole = y.copy()


# In[70]:


whole['ISIG Value'].dropna().count(), whole['Sensor Glucose (mg/dL)'].dropna().count()


# We can perform regression as we have as many ISIG values as Glucose sensor readings. This is however a bit discouraging as it implies that the pump stops logging ISIG values when a calibration deadline is missed, I'm talking from experience.

# In[71]:


"""
bg_idx = whole['BG Reading (mg/dL)'].dropna().index
whole.loc[
    bg_idx - dt.timedelta(minutes=10) : bg_idx + dt.timedelta(minutes=10)
    , 'Sensor Glucose (mg/dL)'
]
"""


# In[72]:


hba1c(whole['Sensor Glucose (mg/dL)'].dropna().mean())


# In[73]:


comparative_hba1c_plot(whole)
dist_plot(whole['Sensor Glucose (mg/dL)'])


# # Last 15 days

# In[74]:


y = whole.loc["2020-03-12":"2020-03-27", :]


# In[75]:


comparative_hba1c_plot(y)
dist_plot(y['Sensor Glucose (mg/dL)'])


# In[76]:


# This is commented out as this function has a bug.
#probability_estimate(y["Sensor Glucose (mg/dL)"], 150, 300, percentage=True)


# In[77]:


#y["Sensor Glucose (mg/dL)"].dropna().apply(int)


# In[78]:


#"dropna" in dir(pd.Series)


# ## Hypoglycaemia pattern detection

# In[79]:


keyword = 'SUSPEND BEFORE LOW'
alarms  = []
for i in y.Alarm.dropna().unique().tolist():
    if keyword in i:
        alarms.append(i)
alarms


# In[80]:


y[ y.Alarm == 'SUSPEND BEFORE LOW ALARM, QUIET' ].hour.hist()


# In[81]:


#meal_id = y['BWZ Carb Input (grams)'].dropna().index
nonull_meals = y['BWZ Carb Input (grams)'].dropna()
nonull_meals = nonull_meals[ nonull_meals > 0 ]
meal_id = nonull_meals.index
print(len(meal_id))
meal_id[:5]


# In[36]:


nonull_corrections = y['BWZ Correction Estimate (U)'].dropna()
nonull_corrections = nonull_corrections[ nonull_corrections > 0 ]
corrections_id = nonull_corrections.index
print(len(corrections_id))
corrections_id[:5]


# In[42]:


bolus_id = corrections_id.union(meal_id)
print(len(bolus_id))


# In[43]:


basal = y.copy()
for uid in bolus_id:
    real = uid+dt.timedelta(hours=2, minutes=30)
    closest = y.index[y.index.searchsorted(real) - 1]  # Otherwise it goes out of bounds !
    basal.loc[uid:closest, 'Sensor Glucose (mg/dL)'] = np.nan


# In[44]:


y.columns


# In[45]:


bolus = pd.DataFrame(columns=['Sensor Glucose (mg/dL)', 'ISIG Value', 'hour', 'd10', 'd20', 'd30', 'x(t)', 'y(t)'], index=y.index)


# In[46]:


for uid in bolus_id:
    real = uid+dt.timedelta(hours=2, minutes=30)
    closest = y.index[y.index.searchsorted(real) - 1]  # Otherwise it goes out of bounds !
    bolus.loc[uid:closest, bolus.columns] = y.loc[uid:closest, bolus.columns]


# In[47]:


#sns.pairplot(bolus.loc[ bolus["Sensor Glucose (mg/dL)"].dropna().index, : ])
#bolus["Sensor Glucose (mg/dL)"].groupby(clean_index.hour).mean().plot()


# In[48]:


y.loc[bolus_id[5], bolus.columns]


# In[49]:


show = False
if show:
    y.loc['2020-03-15', 'Sensor Glucose (mg/dL)'].plot(**{"label": "Full"})
    basal.loc['2020-03-15', 'Sensor Glucose (mg/dL)'].plot(**{"label": "basal"})
    plt.legend()


# In[50]:


#basal.groupby(basal.index.hour)['Sensor Glucose (mg/dL)'].mean().plot()


# In[51]:


figs = [basal.groupby(basal.index.hour)[f'd{i}'].mean().plot(label=f"{i} min") for i in [10, 20, 30]]
figs[-1].legend()


# In[53]:


#help(pd.Series.interpolate)


# In[54]:


methods = [
    'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial',
    'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima'
]


# In[56]:


(
  lambda w: y.loc['2020-03-14 09:00':'2020-03-14 21:00', 'Sensor Glucose (mg/dL)'].interpolate(method=w).plot(label=w)
)(methods[6])
y.loc['2020-03-14 09:00':'2020-03-14 21:00', 'Sensor Glucose (mg/dL)'].plot(label="Original")
plt.legend()


# In[ ]:





# In[57]:


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




