import copy
import pandas as pd

from ..utils.customobjs import Path as path, objdict as odict

from ..assets.carelink_parsing import carelink_parse_kw as parsing


def parse_carelink(handle, addedcols: bool = True):
    """ """
    if addedcols:
        parsing.dtypes.no_sparse.update(parsing.dtypes.added)

    frame = pd.read_csv(handle, **parsing.kwargs)
    for column in frame.iteritems():
        if column[0] in parsing.dtypes.no_sparse.keys():
            frame.loc[:, column[0]] = column[1].astype(parsing.dtypes.no_sparse[column[0]])

    for column in parsing.timedelta.cols:
        frame.loc[:, column] = frame.loc[:, column].fillna("").apply(pd.to_timedelta)

    return frame


if __name__ == "__main__":
    pass
