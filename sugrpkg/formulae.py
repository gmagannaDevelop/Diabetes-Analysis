"""
    Taken from Dr Bode's Prtotocol, published by Medtronic.
"""

########################### IMPORTS #######################################
import math
import numpy as np
import pandas as pd

from typing import Optional, Callable, Any, Dict, Tuple, List, Union, NoReturn

# here defined
from customobjs import objdict

###########################################################################

########################### CONSTANTS #####################################
BASAL_RATIO = 0.5
BOLUS_RATIO = 0.5
###########################################################################


def total_daily_dose_bw(bw: float) -> float:
    """ Compute the total daily dose (tdd) based on body weight (bw) """
    return 0.5 * bw


def total_daily_dose_transition(f_tdd: float) -> float:
    """ Compute the total daily dose (tdd) based on a former
    (pre-insulin-pump) tdd. 
    """
    return 0.75 * f_tdd


def total_basal_dose(tdd: float, basal_ratio: Optional[float] = None) -> float:
    """ Take the basal dose as BASAL_RATIO * tdd
    note that BASAL_RATIO is 0
    """
    basal_ratio = basal_ratio or BASAL_RATIO
    return basal_ratio * tdd


def hourly_basal(tbd: float) -> float:
    """
    """
    return tbd / 24


def carb_to_insulin_ratio_bw(bw: float, tdd: float) -> float:
    """
    """
    return bw * 6 / tdd

def carb_to_insulin_ratio(tdd: float) -> float:
    """ CIR = 450 / TDD """
    return 450 / tdd 

def compute_all_doses_bw(bw: float) -> objdict:
    """
    """
    doses = objdict({
        "TDD": total_daily_dose_bw(bw),
        ""
    })


