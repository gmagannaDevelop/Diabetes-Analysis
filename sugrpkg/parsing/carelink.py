import copy
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from ..utils.customobjs import Path as path, objdict as odict

from ..assets.carelink_parsing import carelink_parse_kw as parsing

BASAL_COL = "Basal Rate (U/h)"


def _ducktape_to_float(column: pd.Series):
    """ """
    return (
        column.apply(lambda x: "" if pd.isna(x) else x)
        .apply(lambda x: x if x.replace(".", "", 1).isdigit() else np.nan)
        .astype(float)
    )


def parse_carelink(handle, addedcols: bool = True):
    """ """
    if addedcols:
        parsing.dtypes.no_sparse.update(parsing.dtypes.added)

    frame = pd.read_csv(handle, **parsing.kwargs)
    if not is_float_dtype(frame[BASAL_COL]):
        frame.loc[:, BASAL_COL] = _ducktape_to_float(frame[BASAL_COL])
    for column in frame.iteritems():
        if column[0] in parsing.dtypes.no_sparse.keys():
            frame.loc[:, column[0]] = column[1].astype(
                parsing.dtypes.no_sparse[column[0]]
            )

    for column in parsing.timedelta.cols:
        frame.loc[:, column] = frame.loc[:, column].fillna("").apply(pd.to_timedelta)

    return frame


if __name__ == "__main__":
    pass
