#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import sys
import toml

from typing import List, Callable, Optional, Dict, NoReturn, Any, Union

import multiprocessing as mp
from functools import reduce, partial

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from preproc import time_indexed_df, new_hybrid_interpolator, merge_on_duplicate_idx
from customobjs import objdict

help(time_indexed_df)


# In[3]:


print(plt.style.available)
styles = plt.style.available


# In[4]:


get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (15, 8)


# In[8]:


x = pd.read_csv(
    "comparative_trial_data/fml.txt",
    sep="\t",
    index_col="Time",
    parse_dates=["Time"],
).drop(["ID"], axis="columns")
x = x.sort_index()
x.index[0], x.index[-1]


# In[9]:


x = x["2020-05-12":"2020-05-25"]
x = merge_on_duplicate_idx(x, verbose=True)


# In[10]:


with open("preproc.toml", "r") as f:
    config = toml.load(f, _dict=objdict)


# In[11]:


x.columns


# In[52]:


start = "2020-05-23"
end = "2020-05-25"
col = "Historic Glucose (mg/dL)"


# In[57]:


config.interpolation.specs


# In[58]:


config.interpolation.specs.limit = 300


# In[59]:


new_hybrid_interpolator(
    x[start:end][col].resample("1T").asfreq(), **config.interpolation.specs
).plot(label="interpolated")
x[start:end][col].plot(label="original")
plt.legend()


# In[60]:


y = x.resample("1T").asfreq()
y[col] = new_hybrid_interpolator(y[col], **config.interpolation.specs)


# In[61]:


y[col].to_csv("faml_interpolated.csv")

sns.distplot(x["Historic Glucose (mg/dL)"])
# In[35]:


# In[18]:


help(pd.DataFrame.drop)


# In[ ]:
