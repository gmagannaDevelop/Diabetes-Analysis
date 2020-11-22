#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Annotations :
from typing import List, Dict, Callable, NoReturn, Any, Optional

# Data and numerical :
import numpy as np
import pandas as pd

# Plotting :
import matplotlib.pyplot as plt
import seaborn as sns

# Local :
from preproc import time_indexed_df, import_csv
from Utils import comparative_hba1c_plot, proportions_visualiser, dist_plot

# Debugging only, remove after building :
get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (15, 8)


# In[9]:


cgm_data = import_csv("preprocessed/CareLink-19-apr-2020-3-months.csv")


# In[2]:


data = pd.read_csv("mySugr_data/Export.csv")
data.columns


# In[3]:


# Date-time indexing :
x = data.copy()
x["DateTime"] = x["Date"] + " " + x["Time"]
x.drop(["Date", "Time"], axis=1, inplace=True)
y = time_indexed_df(x, "DateTime")
y.index = y.index.map(lambda t: t.replace(second=0))


# In[4]:


real = y["2020"]
real.columns


# In[5]:


comparative_hba1c_plot(real, colum_name=real.columns[1])


# In[6]:


proportions_visualiser(real, colum_name=real.columns[1], kind="tar")


# In[7]:


dist_plot(real["Blood Sugar Measurement (mg/dL)"])


# In[10]:


dist_plot(cgm_data["Sensor Glucose (mg/dL)"])


# In[ ]:
