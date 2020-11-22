#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
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
import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing

from typing import List, Dict, NoReturn, Any, Callable, Union, Optional

from preproc import import_csv

import decorators


# In[2]:


@decorators.time_log("logs/timing/interpolation.jl")
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

    quarts = scipy.stats.mstats.mquantiles(series, [0.001, 0.25, 0.5, 0.75, 0.975])

    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}
    )
    sns.boxplot(series, ax=ax_box)
    sns.stripplot(series, color="orange", jitter=0.2, size=2.5, ax=ax_box)
    sns.distplot(series, ax=ax_hist, kde=True)
    ax_hist.axvline(series.mean())
    ax_hist.set_xticks(quarts)
    # ax_box.set(xlabel=f'Mean value : {int(series.mean())}')
    plt.title(
        f"Glycaemic Distribution μ = {int(series.mean())}, σ = {int(series.std())}"
    )


##


@decorators.time_log("logs/timing/interpolation.jl")
def comparative_hba1c_plot(
    df: pd.core.frame.DataFrame,
    colum_name: str = "Sensor Glucose (mg/dL)",
    hba1c: Callable = lambda x: (x + 105) / 36.5,
    windows: Dict[str, int] = {"weekly": 7, "monthly": 30},
    kind: str = "mean",
) -> NoReturn:
    """"""

    glc_to_hba1c = lambda x: (x + 105) / 36.5
    hba1c_to_glc = lambda x: x * 36.5 - 105
    valid_kinds = ["mean", "std", "var"]

    if kind in valid_kinds:
        df.groupby(df.index.dayofyear)[colum_name].apply(eval(f"np.{kind}")).apply(
            hba1c
        ).plot(**{"label": "daily"})

        for key, value in windows.items():
            ax = (
                df.groupby(df.index.dayofyear)[colum_name]
                .apply(eval(f"np.{kind}"))
                .rolling(value)
                .mean()
                .apply(hba1c)
                .plot(**{"label": key})
            )

        ax.set_ylabel("HbA1c %")
        mean_hba1c = glc_to_hba1c(eval(f"df[colum_name].{kind}()"))
        secax = ax.secondary_yaxis("right", functions=(hba1c_to_glc, glc_to_hba1c))
        secax.set_ylabel("mg/dL")
        plt.axhline(
            mean_hba1c, **{"label": f"mean = {round(mean_hba1c,1)}", "c": "blue"}
        )
        plt.legend()
        plt.title(f"Average {kind} of {colum_name}")
    else:
        raise Exception("kind should be `mean` (`std` or `var`)")


##


def nonull_indices(
    df: pd.DataFrame, column: str
) -> pd.core.indexes.datetimes.DatetimeIndex:
    """"""
    _nonull = df[column].dropna()
    _nonull = _nonull[_nonull > 0]
    return _nonull.index


##


def ez_bolus_indices(
    df: pd.DataFrame, kind: Optional[str] = None
) -> pd.core.indexes.datetimes.DatetimeIndex:
    """"""

    _kind_dict = {
        "meal": ["BWZ Carb Input (grams)"],
        "correction": ["BWZ Correction Estimate (U)"],
        "both": ["BWZ Correction Estimate (U)", "BWZ Carb Input (grams)"],
    }
    if kind not in _kind_dict.keys():
        warnings.warn(f"Invalid kind, use of {list(_kind_dict.keys())}.")
        warnings.war("Defaulted to 'both'")
        kind = "both"

    columns = _kind_dict[kind]
    _nonull = partial(nonull_indices, df)
    indices_ls = list(map(_nonull, columns))

    return reduce(lambda x, y: x.union(y), indices_ls)


##


def bolus_indices_explicit(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.core.indexes.datetimes.DatetimeIndex:
    """"""

    columns = columns or ["BWZ Correction Estimate (U)", "BWZ Carb Input (grams)"]
    _nonull = partial(nonull_indices, df)
    indices_ls = list(map(_nonull, columns))
    return reduce(lambda x, y: x.union(y), indices_ls)


##


def basal_only(
    df: pd.DataFrame,
    column: str = "Sensor Glucose (mg/dL)",
    time_window: Dict[str, int] = {"hours": 2, "minutes": 30},
) -> pd.DataFrame:
    """"""
    basal = df.copy()
    _td = dt.timedelta(**time_window)
    for uid in bolus_indices(basal):
        real = uid + _td
        closest = df.index[
            df.index.searchsorted(real) - 1
        ]  # Otherwise it goes out of bounds !
        basal.loc[uid:closest, column] = np.nan
    return basal


##


def hourly_trends(df: pd.DataFrame, kind: str = "mean") -> NoReturn:
    """"""
    valid_kinds = ["mean", "std", "var"]

    if kind in valid_kinds:
        figs = [
            df.groupby(df.index.hour)[f"d{i}"]
            .apply(eval(f"np.{kind}"))
            .plot(label=f"{i} ")
            for i in [10, 20, 30]
        ]
        figs[-1].legend()
        plt.title(f"Hourly trends : {kind}")
        plt.xticks([i for i in range(24)])
        plt.ylabel("mg/dl")
    else:
        raise Exception(f"Invalid kind, select one from {valid_kinds}")


##


# In[3]:


random_seed = 123456


# In[4]:


get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["figure.max_open_warning"] = False


# In[5]:


data = import_csv("preprocessed/CareLink-23-apr-2020-1-month.csv")


# In[6]:


print("start \t:", data.index[0])
print("end \t:", data.index[-1])


# In[7]:


latest = data.loc["2020-04-19":"2020-04-23", :]


# In[8]:


# help(pd.plotting.autocorrelation_plot)


# In[9]:


for i in range(1, 10):
    plt.figure()
    pd.plotting.lag_plot(latest["Sensor Glucose (mg/dL)"], lag=i)


# **Regular**
#
# ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’
#
# **Special**
#
# ‘krogh’, ‘piecewise_polynomial’, ‘spline’, ‘pchip’, ‘akima’

# In[10]:


def new_hybrid_interpolator(
    data: pd.core.series.Series,
    methods: Dict[str, float] = {"linear": 0.65, "spline": 0.35},
    direction: str = "forward",
    limit: int = 120,
    limit_area: Optional[str] = None,
    order: int = 2,
    **kw,
) -> pd.core.series.Series:
    """"""

    limit_area = limit_area or "inside"

    weight_sum = sum(weight for weight in methods.values())
    if not np.isclose(weight_sum, 1):
        raise Exception(f"Sum of weights {weight_sum} != 1")

    resampled: Dict[str, pd.core.series.Series] = {}

    for key, weight in methods.items():
        resampled.update(
            {
                key: weight
                * data.resample("1Min").interpolate(
                    method=key, order=order, limit_area=limit_area, limit=120
                )
            }
        )

    return reduce(lambda x, y: x + y, resampled.values())


##


# In[11]:


plt.close("all")


# In[12]:


# data[ data.index.month == 4 ].loc["Sensor Glucose (mg/dL)"].plot()


# In[13]:


pd.concat([data["2020-04-05"], data["2020-04-06"]]).index


# In[14]:


@decorators.time_log("logs/timing/interpolation.jl")
def full_direct_interpolation(df: pd.DataFrame, **kw) -> pd.Series:
    """"""
    return new_hybrid_interpolator(
        df["Sensor Glucose (mg/dL)"],
        methods={"linear": 0.65, "akima": 0.25, "polynomial": 0.10},
        direction="both",
    )


##


@decorators.time_log("logs/timing/interpolation.jl")
def groupby_interpolation(df: pd.DataFrame, **kw) -> pd.Series:
    """"""

    interpolated_ls: List[pd.Series] = []

    for day, frame in df.groupby(df.index.date):
        interpolated_ls.append(
            new_hybrid_interpolator(
                frame["Sensor Glucose (mg/dL)"],
                methods={"linear": 0.65, "akima": 0.25, "polynomial": 0.10},
                direction="both",
            )
        )

    return pd.concat(interpolated_ls)


##


# In[15]:


woo = sorted(list(map(str, set(data.index.date))))


# In[16]:


len(woo)


# In[17]:


woo_dict = {i: woo[0 : i + 1] for i in range(len(woo))}


# In[18]:


for i in range(1, len(woo)):
    _ = full_direct_interpolation(
        data.loc[woo[0] : woo[i], :],
        days=i,
    )
    __ = groupby_interpolation(data.loc[woo[0] : woo[i], :], days=i)


# In[20]:


#%%timeit
full_interpolated = new_hybrid_interpolator(
    data["Sensor Glucose (mg/dL)"],
    methods={"linear": 0.65, "akima": 0.25, "polynomial": 0.10},
    direction="both",
    comment="dub dub da dup doop",
)


# In[19]:


463 / 69.3


# In[21]:


#%%timeit
interpolated_ls: List[pd.Series] = []
for day, frame in data.groupby(data.index.date):
    interpolated_ls.append(
        new_hybrid_interpolator(
            frame["Sensor Glucose (mg/dL)"],
            methods={"linear": 0.65, "akima": 0.25, "polynomial": 0.10},
            direction="both",
        )
    )

interpolated: pd.Series = pd.concat(interpolated_ls)


# In[22]:


plt.close("all")
for date in map(str, sorted(set(data.index.date))):
    plt.figure()
    full_interpolated[date].plot(**{"label": "fully interpolated"})
    interpolated[date].plot(**{"label": "interpolated by day"})
    data.loc[date, "Sensor Glucose (mg/dL)"].plot(**{"label": "original"})
    plt.legend()


# In[ ]:


# In[41]:


sum(
    latest["Sensor Glucose (mg/dL)"].dropna()[0:3],
    latest["Sensor Glucose (mg/dL)"].dropna()[0:3],
)


# In[17]:


interpolators = {
    "linear": "blue",
    "akima": "green",
    # "spline": "yellow"
}


# In[22]:


for day, frame in data.groupby(data.index.day):
    plt.figure()
    """
    for met, col in interpolators.items():
        frame["Sensor Glucose (mg/dL)"].resample("1Min").\
            interpolate(
                method=met,
                order=2,
                limit=120
            ).plot(
                **{"color": col, "label": met}
            )
    """

    """
    frame["Sensor Glucose (mg/dL)"].\
        rolling(
            window=3,
            center=True
        ).mean().resample("1Min").interpolate().plot(**{"color": "brown", "label": "rolling"})
    """

    sns.scatterplot(
        x=frame.index,
        y="Sensor Glucose (mg/dL)",
        data=frame,
        **{"color": "orange", "marker": "d"},
    )

    # frame["Sensor Glucose (mg/dL)"].resample("1Min").fillna("").plot()


# In[33]:


"pad" in dir(pd.DataFrame.resample)


# In[102]:


idx = data.index.copy()


# In[103]:


def bolus_only(
    df: pd.DataFrame,
    column: str = "Sensor Glucose (mg/dL)",
    time_window: Dict[str, int] = {"hours": 2, "minutes": 30},
    kind: Optional[str] = None,
) -> pd.DataFrame:
    """
    kind: 'meal'
          'correction'
          'both'

           defaults to 'both'
    """
    bolus = df.copy()
    bolus["save"] = False

    _kind_dict = {
        "meal": ["BWZ Carb Input (grams)"],
        "correction": ["BWZ Correction Estimate (U)"],
        "both": ["BWZ Correction Estimate (U)", "BWZ Carb Input (grams)"],
    }
    if kind not in _kind_dict.keys():
        warnings.warn(f"Invalid kind, use of {list(_kind_dict.keys())}.")
        warnings.war("Defaulted to 'both'")
        kind = "both"

    _td = dt.timedelta(**time_window)

    for uid in bolus_indices_explicit(bolus, columns=_kind_dict[kind]):
        real = uid + _td
        closest = df.index[
            df.index.searchsorted(real) - 1
        ]  # Otherwise it goes out of bounds !
        bolus.loc[uid:closest, "save"] = True

    bolus.loc[bolus.save == False, "Sensor Glucose (mg/dL)"] = np.nan

    return bolus


##


def overlapping_dayplot(
    df: pd.DataFrame,
    sns_scatter_kw: Dict[str, Any] = {
        "x": "minutes",
        "y": "Sensor Glucose (mg/dL)",
    },
) -> NoReturn:
    """"""
    sns.scatterplot(data=df, hue=df.index.date, **sns_scatter_kw)


##


# In[56]:


data.columns


# In[124]:


las_columnas = [
    "Basal Rate (U/h)",
    "Sensor Glucose (mg/dL)",
    "Bolus Volume Delivered (U)",
]


# In[ ]:


# In[125]:


np.isnan(np.nan)


# In[126]:


latest[las_columnas[-1]].index


# In[127]:


# latest[las_columnas[1]].fillna(method="ffill", inplace=True)


# In[ ]:


# In[128]:


latest.loc[:, las_columnas].plot(subplots=True, kind="line")


# In[61]:


bolos = bolus_only(
    latest,
)


# In[76]:


comidas = bolus_only(latest, kind="meal")


# In[106]:


_k_means = (
    lambda x: KMeans(n_clusters=3, random_state=random_seed, verbose=True).fit(
        x[["x(t)", "y(t)"]]
    )
)(comidas.loc[ez_bolus_indices(comidas, kind="meal"), :])


# In[108]:


comidas.loc[ez_bolus_indices(comidas, kind="meal"), "labels"] = _k_means.labels_


# In[113]:


comidas.labels.fillna(method="ffill", inplace=True)


# In[117]:


comidas.loc[np.isnan(comidas["Sensor Glucose (mg/dL)"]), "labels"] = np.nan


# In[119]:


overlapping_dayplot(comidas[comidas.labels == 0])


# In[83]:


help(KMeans)


# In[79]:


correcciones = bolus_only(latest, kind="correction")


# In[64]:


basal = basal_only(latest)


# In[65]:


bolos.loc[bolos.save == False, "Sensor Glucose (mg/dL)"] = np.nan


# In[86]:


overlapping_dayplot(comidas)


# In[80]:


overlapping_dayplot(correcciones)


# In[69]:


overlapping_dayplot(basal)


# In[63]:


sns.scatterplot(
    data=bolos,
    x="minutes",
    y="Sensor Glucose (mg/dL)",
    hue=bolos.index.date,
    size="d30",
)


# In[10]:


# dir(idx)


# In[9]:


help(idx.drop)


# In[ ]:
