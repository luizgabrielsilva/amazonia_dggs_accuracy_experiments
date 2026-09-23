"""

Define functions for running the experiments


"""


# ----------------------------------------------------------------------------
# SetUp

import os
import sys

PROJPATH = '/home/jovyan/projetos/proj_amazonia_dggs'
PROJBASEPATH = f'{PROJPATH}/proj_amazonia_dggs_base'

ENVPATH = f'{PROJBASEPATH}/env/proj_amazonia_dggs'
SRCPATH = f'{PROJBASEPATH}/src'

LOCALPATH = f'{PROJPATH}/amazonia_dggs_accuracy_experiments'
DATAPATH = f'{LOCALPATH}/data'
DOCPATH = f'{LOCALPATH}/docs'

os.environ['PROJ_DATA'] = f'{ENVPATH}/share/proj'
sys.path.append(SRCPATH)

# ----------------------------------------------------------------------------
# Packages

from typing import List
from numpy.random._generator import Generator

import numpy as np
import pandas as pd


from scipy.stats import bootstrap
from tqdm.notebook import tqdm

# ----------------------------------------------------------------------------
# Local Packages

import area_definitions

# ----------------------------------------------------------------------------
# Global variables (include for convenience)

AREA_SCALE_FACTOR: int = area_definitions.AREA_SCALE_FACTOR
DEFALT_RNG: Generator = np.random.default_rng(2026)

# ----------------------------------------------------------------------------
# Functions

def __compute(x: np.ndarray) -> np.ndarray :
    return np.sqrt(np.mean(x))


def get_rmse(exp_data: pd.DataFrame,
             col_names: List[str],
             n_resamples: int = 1000,
             confidence_level: float = 0.95,
             rng: Generator = DEFALT_RNG):
    
    rmse: List[float] = []
    low_ic: List[float] = []
    high_ic: List[float] = []
    
    for c in tqdm(col_names):
        data = (exp_data['area'] - exp_data[c]).values ** 2
    
        res = bootstrap((data,), __compute,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples, rng=rng)
        rmse.append(__compute(data))
        low_ic.append(res.confidence_interval.low)
        high_ic.append(res.confidence_interval.high)

    return rmse, low_ic, high_ic


def compute_rel_error(data:pd.DataFrame,
                      threshold: float = '0.5') -> np.ndarray:
    rel_error = ((data['area'] -  data[f'area_{threshold}']).values
                 / data['area'].values)
    return rel_error


def get_rel_error_shape_metrics(
    exp_data: pd.DataFrame,
    categories: List[str],
    shape_metric_name: str,
    threshold: float = '0.5',
    ):

    data = exp_data.copy()
    data['rel_error'] = compute_rel_error(data, threshold=threshold)
    

    rel_error=[]
    n = []
    shape_metric = []
    for cat in categories:
        tmp = data.loc[data.category == cat]
        rel_error.append(tmp['rel_error'].values)
        shape_metric.append(tmp[shape_metric_name].values)
        n.append(tmp[f'n_{threshold}'].values)

    return rel_error, shape_metric, n
