"""
    Preprocessing
"""

import os
import io
import sys
import multiprocessing as mp
from functools import reduce, partial

import toml

import pandas as pd
import numpy as np
import datetime as dt

import copy
import gc

from typing import List, Dict, NoReturn, Any, Callable, Union, Optional

# Local imports
from decorators import time_log, time_this
from customobjs import objdict

@time_log("logs/preprocessing.jsonl")
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

@time_log("logs/preprocessing.jsonl")
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

@time_log("logs/preprocessing.jsonl")
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

def import_csv(
    filename: str,
    column: Optional[str] = None
) -> pd.DataFrame:
    """
        Import a csv file previously generated with this same script.
        
        `column` optional parameter specifies the 
                 name of the column containing 
                 the datetime index.
                 Defaults to 'DateTime'
    """

    column = column or "DateTime"
    _x = pd.read_csv(filename)
    _y = time_indexed_df(_x, column)

    return _y
##

@time_this
def grep_idx(file_lines: List[str], the_string: str) -> List[int]: 
    """
        Function definition inpired by Dorian Grv :
        https://stackoverflow.com/users/2444948/dorian-grv, 
        
        Originally used re.search in the context of this question:
        https://stackoverflow.com/questions/4146009/python-get-list-indexes-using-regular-expression
    """

    ide = [i for i, item in enumerate(file_lines) if item.startswith(the_string)] 
    return ide 



@time_this
@time_log("logs/preprocessing.jsonl")
def main(config_file: str) -> NoReturn:
    """
    
    """
    
    # Parse config :
    with open(config_file, "r") as f:
        config = toml.load(config_file, _dict=objdict)


    get_csv_files = lambda loc: [os.path.join(loc, x) for x in os.listdir(loc) if x[-4:] == ".csv"] 
    
    config.files = { 
        key: get_csv_files(value) 
        for key, value in config.locations.items()
    }

    # Dump machine-generated conf
    with open("auto-config.toml", "w") as f:
        toml.dump(config, f)
    sys.exit()

    with open(in_file, "r") as f:
        f_list = f.readlines()[6:]
        idx_to_remove = grep_idx(f_list, "-------,")
        _tmp_ixs = []
        # Quadratic complexity :(
        for ix in idx_to_remove:
            _tmp_ixs.extend([ix-1, ix+1])
        idx_to_remove.extend(
            [i for i in _tmp_ixs if i in range(len(f_list))]
        )

        for i in reversed(sorted(idx_to_remove)):
            _ = f_list.pop(i)
    
    f_str = "".join(f_list)
    
    with io.StringIO(f_str) as g:
        x = pd.read_csv(g)
    
    # Date-time indexing :
    x["DateTime"] =  x["Date"] + " " + x["Time"]
    x.drop(["Date", "Time"], axis=1, inplace=True)
    y = time_indexed_df(x, 'DateTime')
    y.drop("Index", axis=1, inplace=True)
    y.index = y.index.map(lambda t: t.replace(second=0))
   
    # Merge duplicates :
    y = merge_on_duplicate_idx(y, verbose=True)
    
    # Useful having an hour column, for groupby opperations :
    y['hour'] = y.index.hour
    # Deltas within valuable intervals : 
    for i in [10, 20, 30]: 
        y[f'd{i}'] = y['Sensor Glucose (mg/dL)'].diff(i)
    
    # Coulmns to capture daily periodicity :
    T = 1439
    min_res_t_series = pd.Series(y.hour*60 + y.index.minute)
    y['minutes'] = min_res_t_series
    y['x(t)'] = min_res_t_series.apply(lambda x: np.cos(2*np.pi*(x) / T))
    y['y(t)'] = min_res_t_series.apply(lambda x: np.sin(2*np.pi*(x) / T))

    # Fill missing basal values :
    y["Basal Rate (U/h)"].fillna(method="ffill", inplace=True)
    
    y.to_csv(out_file)
##

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: \n $ python {sys.argv[0]} config_file.toml")
        print(f"config_file.toml must be in toml format ! https://github.com/toml-lang/toml")
        exit()
    else:
        main(sys.argv[1])
##
