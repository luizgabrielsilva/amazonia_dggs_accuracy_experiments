"""

Define functions for sampling the set of polygons and for computing shape
metrics.

Computed metrics:
1) Maximum Inscribed Circle Diameter (MICD)
2) Boundary Amplitude (BA)
3) Squareness
4) Rectangularity
5) Minimum Bounding Circle Ratio (MBCR)

"""


# ----------------------------------------------------------------------------
# SetUp

import os
import sys

PROJPATH = '/home/jovyan/projetos/proj_amazonia_dggs'
PROJBASEPATH = f'{PROJPATH}/proj_amazonia_dggs_base'

ENVPATH = f'{PROJBASEPATH}/env/proj_amazonia_dggs'
SRCPATH = f'{PROJBASEPATH}/scr'

LOCALPATH = f'{PROJPATH}/amazonia_dggs_accuracy_experiments'
DATAPATH = f'{LOCALPATH}/data'
DOCPATH = f'{LOCALPATH}/docs'

os.environ['PROJ_DATA'] = f'{ENVPATH}/share/proj'
sys.path.append(SRCPATH)


# ----------------------------------------------------------------------------
# Packages

from typing import NoReturn

import geopandas as gpd
from esda import shape as shapestats

# ----------------------------------------------------------------------------
# Local Packages

import crs_definitions


# ----------------------------------------------------------------------------
# Global variables (include for convenience)



# ----------------------------------------------------------------------------
# Functions


def get_data(path: str,
             layer:str, 
             area_threshold: float = 0.05) -> gpd.GeoDataFrame:

    data = gpd.read_file(path, layer=layer)

    data = data.loc[data['AREA_Km²'] >= area_threshold]

    data = (
        data
            .drop(columns=['VALUE', 'CLASSE', 'ANO', 'AREA_ha'])
            .rename(columns={'AREA_Km²': 'area'})
)

    return data


def get_sample(data: gpd.GeoDataFrame,
               frac: float = 0.05,
               random_state: int = 2026) -> gpd.GeoDataFrame:
    
    data = (
        data
            .drop(columns=['area'])       
            .sample(frac=frac, random_state=random_state)
    )

    return data


def compute_metrics(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data_projected = data.to_crs(crs_definitions.SOUTH_AMERICA_LAMBERT)

    data_projected['mbcr'] = shapestats.minimum_bounding_circle_ratio(
        data_projected.geometry.values)[0]

    data_projected['ba'] = shapestats.boundary_amplitude(
        data_projected.geometry.values)[0]

    data_projected['squareness'] = shapestats.squareness(
        data_projected.geometry.values)[0]

    data_projected['rectangularity'] = shapestats.rectangularity(
        data_projected.geometry.values)[0]

    data_projected['micd'] = (
        2 * data_projected.maximum_inscribed_circle().length / 1e3
    )

    data_projected = data_projected[['mbcr', 'ba', 'squareness',
                                     'rectangularity', 'micd']]

    data = data.merge(data_projected,
                      right_index=True,
                      left_index=True,
                      how='inner')

    return data


def main(input_path: str,
         input_layer_name: str,
         output_path: str) -> NoReturn:
    
    # Reading and Filtering data
    data = get_data(path=input_path, layer=input_layer_name)

    # Sampling data
    data = get_sample(data=data)

    # Computing metrics
    data = compute_metrics(data)

    # Saving data
    data.to_parquet(output_path, index=False)


if __name__ == '__main__':
    input_path = f'{DATAPATH}/raw/VS_Amazônia_2022.gpkg'
    input_layer_name = 'VS_Amazonia_2022'
    output_path = f'{DATAPATH}/processed/sample/sample_with_metrics.parquet'

    main(
        input_path=input_path,
        input_layer_name=input_layer_name,
        output_path=output_path
    )
