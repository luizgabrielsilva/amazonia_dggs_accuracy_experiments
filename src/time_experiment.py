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

from typing import List, Dict, NoReturn

import time

import geopandas as gpd
import numpy as np
import pandas as pd


from tqdm import tqdm

# ----------------------------------------------------------------------------
# Local Packages

import crs_definitions
import dggal_utils
import rhealpix_utils

# ----------------------------------------------------------------------------
# Global variables (include for convenience)


# ----------------------------------------------------------------------------
# Functions


def get_time_experiment_data(ivea7h, rhealpix_dggal, rhealpix, polygons):
    times_ivea7h = []
    times_rhealpix_dggal = []
    times_rhealpix = []
    
    for _, row in tqdm(polygons.iterrows(), total=polygons.shape[0]):
        
        # Initialize variables
        geometry = row.geometry

        # cover geometry with cells
        start_time = time.perf_counter()
        data_ivea7h = dggal_utils.get_grid_from_geometry(
            geometry=geometry,
            dggs=ivea7h,
            level=11,
        )
        end_time = time.perf_counter()
        times_ivea7h.append(end_time - start_time)

        start_time = time.perf_counter()
        data_rhealpix_dggal = dggal_utils.get_grid_from_geometry(
            geometry=geometry,
            dggs=rhealpix_dggal,
            level=10,
        )
        end_time = time.perf_counter()
        times_rhealpix_dggal.append(end_time - start_time)

        start_time = time.perf_counter()
        data = rhealpix_utils.get_grid_from_geometry(
            geometry=geometry,
            dggs=rhealpix,
            level=10,
        )
        end_time = time.perf_counter()
        times_rhealpix.append(end_time - start_time)

    result = polygons[['area']].copy()
    result['time_ivea7h'] = times_ivea7h
    result['time_rhealpix_dggal'] = times_rhealpix_dggal
    result['time_rhealpix'] = times_rhealpix

    return result


def main(input_path: str, output_folder: str) -> NoReturn:
    polygons = gpd.read_parquet(input_path).to_crs(crs_definitions.WGS84)

    exp_data = get_time_experiment_data(
        ivea7h=dggal_utils.IVEA7H,
        rhealpix_dggal=dggal_utils.RDGGS3,
        rhealpix=rhealpix_utils.RDGGS3,
        polygons=polygons,
    )

    exp_data.to_parquet(
        f'{output_folder}/time_experiment.parquet'
    )


if __name__ == '__main__':
    input_path = f'{DATAPATH}/processed/sample/sample_with_metrics.parquet'
    output_folder = f'{DATAPATH}/processed/experiments'

    main(
        input_path=input_path,
        output_folder=output_folder
    )

