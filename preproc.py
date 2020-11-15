"""
    Preprocessing
"""

# System and io :
import os
import io
import sys
import toml
import copy
import gc

# Functional and multiprocessing
# (elegance and performance) :
import multiprocessing as mp
from concurrent import futures
import threading
from functools import reduce, partial

# Numerical and data :
import pandas as pd
import numpy as np
import datetime as dt

# Type hints :
from typing import List, Dict, NoReturn, Any, Callable, Union, Optional

# Local imports
from decorators import time_log, time_this
from customobjs import objdict
################################################################################


def exit_explain_usage():
    """
        Pretty self explanatory, isn't it ?
    """
    print(f"\n\nUsage: \n $ python {sys.argv[0]} config_file")
    print(f"Config file should be .toml format : https://github.com/toml-lang/toml\n\n")
    exit()
##

def is_non_zero_file(fpath):
    """
    """
    try:
        return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    except:
        return False
##

def parse_config(config_file: str) -> objdict:
    """
    """
    try:
        with open(config_file, "r") as f:
            _config = toml.load(f, _dict=objdict)
        return _config
    except:
        raise
##

def update_lock(lock_file: str, data: objdict) -> NoReturn:
    """
    """
    try:
        with open(lock_file, "w") as f:
            toml.dump(data, f)
    except:
        raise
##

#@time_log("logs/preprocessing.jsonl")
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

def new_hybrid_interpolator(
    data: pd.core.series.Series,
    methods: Dict[str,float] = {
        'linear': 0.65,
        'spline': 0.35
    },
    direction: str = 'forward',
    limit: int = 120,
    limit_area: Optional[str] = None,
    order: int = 2,
    **kw
) -> pd.core.series.Series:
    """
    """

    limit_area = limit_area or 'inside'

    weight_sum = sum(weight for weight in methods.values())
    if not np.isclose(weight_sum, 1):
        raise Exception(f'Sum of weights {weight_sum} != 1')

    resampled: Dict[str,pd.core.series.Series] = {}

    for key, weight in methods.items():
        resampled.update({
          key: weight * data.interpolate(
              method=key,
              order=order,
              limit_area=limit_area,
              limit=limit
          )
        })

    return reduce(lambda x, y: x+y, resampled.values())
##

def add_time_periodicity(df: pd.DataFrame) -> NoReturn:
    """
    """
    # Coulmns to capture daily periodicity :
    T = 1439
    min_res_t_series = pd.Series(df.index.hour*60 + df.index.minute)
    df['hour'] = df.index.hour
    df['minutes'] = min_res_t_series
    df['x(t)'] = min_res_t_series.apply(lambda x: np.cos(2*np.pi*(x) / T))
    df['y(t)'] = min_res_t_series.apply(lambda x: np.sin(2*np.pi*(x) / T))
##

def compute_time_periodicity(df: pd.DataFrame) -> NoReturn:
    """
    """
    # Coulmns to capture daily periodicity :
    T = 1439
    min_res_t_series = pd.Series(df.index.hour*60 + df.index.minute)
    _tmp = pd.DataFrame({
        'hour': df.index.hour,
        'minute': min_res_t_series,
        'x(t)': min_res_t_series.apply(lambda x: np.cos(2*np.pi*x / T)),
        'y(t)': min_res_t_series.apply(lambda x: np.sin(2*np.pi*x / T))
    })
    _tmp.index = df.index
    return _tmp
##

def import_csv(
    filename: str,
    column: Optional[str] = None,
    low_memory: bool = False
) -> pd.DataFrame:
    """
        Import a csv file previously generated with this same script.

        `column` optional parameter specifies the
                 name of the column containing
                 the datetime index.
                 Defaults to 'DateTime'
    """

    column = column or "DateTime"
    _x = pd.read_csv(filename, low_memory=low_memory)
    _y = time_indexed_df(_x, column)

    return _y
##

def grep_idx(file_lines: List[str], the_string: str) -> List[int]:
    """
        Function definition inpired by Dorian Grv :
        https://stackoverflow.com/users/2444948/dorian-grv,

        Originally used re.search in the context of this question:
        https://stackoverflow.com/questions/4146009/python-get-list-indexes-using-regular-expression
    """

    ide = [i for i, item in enumerate(file_lines) if item.startswith(the_string)]
    return ide

@time_log("logs/preprocessing.jsonl")
def preproc(config: objdict, file: str) -> NoReturn:
    """
        Congfig file

        this should eventually be renamed to preproc medtronic

        From the first lines of the file we can get this kind of dict :
        {
            'Last Name': 'Maganna',
            'First Name': 'Gustavo',
            'Patient ID': nan,
            'Start Date': '19/04/20 12:00:00 AM',
            'End Date': '16/05/20 11:59:59 PM',
            'Device': 'Serial Number',
            'MiniMed 640G': 'NG1988812H',
            ' MMT-1512/1712': nan
        }

    """
    in_file = os.path.join(config.locations.source, file)
    with open(in_file, "r") as f:
        # Get file information (_fi) :
        _fi = pd.read_csv(f, nrows=1).squeeze().to_dict()

    with open(in_file, "r") as f:
        # Read the actual data :
        f_list = f.readlines()[config.file.specs.header_row_num-1:]

    idx_to_remove = grep_idx(f_list, "-------,")
    _tmp_ixs = []
    # Quadratic complexity :(
    for ix in idx_to_remove:
        #print("remove")
        _tmp_ixs.extend([ix-1, ix+1])
    idx_to_remove.extend(
        [i for i in _tmp_ixs if i in range(len(f_list))]
    )
    # If we didn't sort and reverse, we'd be modifying the list's index order
    # i.e. not removing the desired elements but their neighbours instead.
    for i in reversed(sorted(idx_to_remove)):
        _ = f_list.pop(i)

    f_str = "".join(f_list)

    with io.StringIO(f_str) as g:
        x = pd.read_csv(g, low_memory=False)

    # Date-time indexing :
    x["DateTime"] = x[config.file.specs.date] + " " + x[config.file.specs.time]
    x = x.drop([config.file.specs.date, config.file.specs.time], axis=1)
    y = time_indexed_df(x, 'DateTime')
    if config.file.specs.dummy_index:
        y = y.drop(config.file.specs.dummy_index, axis=1)
    y.index = y.index.map(lambda t: t.replace(second=0))

    # Merge duplicates :
    y = merge_on_duplicate_idx(y, verbose=config.specs.verbose)

    if config.tasks.interpolate:
        y = y.resample("1T").asfreq()
        y[config.file.specs.glycaemia_column] = new_hybrid_interpolator(
            y[config.file.specs.glycaemia_column],
            **config.interpolation.specs
        )
        # Only differentiate if we have interpolated, the definition won't be valid
        # as we need a constant, evenly-spaced temporal grid.
        if config.tasks.differentiate:
            _d = config.differentiation.specs.delta
            _w = config.differentiation.specs.window_size
            y[f"d{_d}w{_w}"] = y[config.file.specs.glycaemia_column].diff(_d).rolling(_w).median()
            y[f"Sd{_d}w{_w}"] = y[config.file.specs.glycaemia_column].diff(_d).rolling(_w).sum()

    if config.tasks.interpolate_isig:
        y[config.file.specs.isig_column] = new_hybrid_interpolator(
            y[config.file.specs.isig_column],
            **config.interpolation.specs
        )

    periodicity_df = compute_time_periodicity(y)
    y = y.join(periodicity_df)

    # Fill missing basal values :
    y["Basal Rate (U/h)"].fillna(method="ffill", inplace=True)

    get_date = lambda x: x.split(" ")[0].replace("/", "-")
    _st, _end = map(get_date, [_fi['Start Date'], _fi['End Date']])
    reverse_date = lambda x: "-".join(list(reversed(x.split("-")))) 
    uniform_date = lambda x: f"0{int(x)}" if int(x) < 10 else x
    _st, _end = map(reverse_date, [_st, _end])
    uniform_dates = lambda y: "-".join(map(uniform_date, y.split("-")))
    _st, _end = map(uniform_dates, [_st, _end])
    out_file = [
        f"{_fi['MiniMed 640G']}", f"{_fi['Last Name']}",
        f"{_fi['First Name']}", 
        f"(til:{_end})", f"(from:{_st})"
    ]
    out_file = "_".join(out_file)
    if config.tasks.interpolate:
        out_file += "_interpolated.csv"
        out_file = os.path.join(config.locations.interpolated, out_file)
    else:
        out_file += ".csv"
        out_file = os.path.join(config.locations.preprocessed, out_file)

    y.to_csv(out_file)
##

def main(config_file: str) -> NoReturn:
    """
    """
    global DEFAULT_LOCK_FILE
    if is_non_zero_file(DEFAULT_LOCK_FILE):
        history = parse_config(DEFAULT_LOCK_FILE)
    else:
        history = objdict({
            "files": objdict({
                "processed": []
            })
        })
        update_lock(DEFAULT_LOCK_FILE, history)

    config = parse_config(config_file)
    #get_csv_files = lambda loc: [os.path.join(loc, x) for x in os.listdir(loc) if x[-4:] == ".csv"]
    get_csv_files = lambda loc: [x for x in os.listdir(loc) if x[-4:] == ".csv"]

    csv_files = get_csv_files(config.locations.source)

    if config.specs.ignore_lock:
        files_to_process = csv_files
    else:
        files_to_process = list(set(csv_files) - set(history.files.processed))

    if files_to_process:
        for task, val in config.tasks.items():
            print(f"{task}\t:\t{val}")
        print(f"\n{len(files_to_process)} files to process : ")
        for file in files_to_process:
            print(f"\t{file}")
        _preproc = partial(preproc, config)
        # DO NOT USE ASYNC UNTIL THE CODE IS WORKING
        if config.tasks.debug:
            for file in files_to_process:
                _preproc(file)
        else:
            with futures.ThreadPoolExecutor(max_workers=config.hardware.n_threads) as pool:
                pool.map(_preproc, files_to_process)
        #list(map(_preproc, files_to_process))
        # If the execution arrived to this point, we can safely
        # say that we've preprocessed the specified files
        history.files.processed += files_to_process
        history.files.processed = list(set(history.files.processed))
        update_lock(DEFAULT_LOCK_FILE, history)
    else:
        print("No files to process, please verify the following :")
        print(f" CONFIG : {config_file} ")
        print(f" LOCK :  {DEFAULT_LOCK_FILE} ")
        print(f" Source directory (parsed from CONFIG) : {config.locations.source}/")
##


if __name__ == "__main__":

    # These DEFAULT FILES ARE DEFINED AT TOP LEVEL
    global DEFAULT_CONFIG_FILE
    global DEFAULT_LOCK_FILE
    DEFAULT_CONFIG_FILE = sys.argv[0].replace(".py", ".toml")
    DEFAULT_LOCK_FILE = sys.argv[0].replace(".py", "_lock.toml")

    if len(sys.argv) != 2:
        if DEFAULT_CONFIG_FILE in os.listdir("."):
            print(f"Config file not specified, using default `{DEFAULT_CONFIG_FILE}`")
            main(DEFAULT_CONFIG_FILE)
        else:
            print(f"Config file not specified, using default `{DEFAULT_CONFIG_FILE}`...")
            print(f"Default `{DEFAULT_CONFIG_FILE}` not found in {os.path.abspath('.')}")
            exit_explain_usage()
    else:
        if "toml" in sys.argv[1]:
            main(sys.argv[1])
        else:
            exit_explain_usage()
##
