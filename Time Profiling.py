#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.max_open_warning'] = False


# In[5]:


ls logs/timing/


# In[17]:


data = pd.read_json("logs/timing/interpolation.jl", lines=True)
data.columns


# In[23]:


for function, frame in data.groupby(data.function):
    frame["execution time (s)"].plot(**{"label": function})
    
plt.legend()


# In[9]:


#with open("logs/timing/interpolation.jl") as f:
    


# In[ ]:




