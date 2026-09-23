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
from itertools import product
import time

import geopandas as gpd
import numpy as np
import pandas as pd

import shapely
from pyproj import Transformer

from tqdm import tqdm

# ----------------------------------------------------------------------------
# Local Packages

import crs_definitions
import area_definitions
import dggal_utils
import rhealpix_utils

# ----------------------------------------------------------------------------
# Global variables (include for convenience)

AREA_SCALE_FACTOR = area_definitions.AREA_SCALE_FACTOR

TRANSFORMER = Transformer.from_crs(crs_definitions.WGS84,
                                       crs_definitions.BRAZIL_ALBERS,
                                       always_xy=True)

# ----------------------------------------------------------------------------
# Functions

def get_overlay(data: gpd.GeoDataFrame,
                cell_area: float,
                roi) -> gpd.GeoDataFrame:

    data['prop'] = (
        data.intersection(roi)
            .to_crs(crs_definitions.BRAZIL_ALBERS)
            .area.values) / (cell_area * AREA_SCALE_FACTOR)

    return data


def __get_children(dggs_name,
                   dggs_instance,
                   parents_ids,
                   level) -> gpd.GeoDataFrame:
    
    if dggs_name == 'IVEA7H':
        data = dggal_utils.get_children(
            dggs=dggs_instance,
            parents_ids=parents_ids,
            aperture=7,
        )
    else:
        data = rhealpix_utils.get_children(
            dggs=dggs_instance,
            parents_ids=parents_ids,
            level=level,
            include_parent_id=False,
        )

    return data


def __get_grid_from_geometry(dggs_name,
                             dggs_instance,
                             geometry,
                             level) -> gpd.GeoDataFrame:
    
    if dggs_name == 'IVEA7H':
        data = dggal_utils.get_grid_from_geometry(
            geometry=geometry,
            dggs=dggs_instance,
            level=level,
        )
    else:
        data = rhealpix_utils.get_grid_from_geometry(
            geometry=geometry,
            dggs=dggs_instance,
            level=level,
    )

    return data


def __get_geom_representation(nested_grid,
                              dggs_name,
                              dggs_instance,
                              level):
    
    if dggs_name == 'IVEA7H':
        uncompacted_ids = dggal_utils.uncompact_cells(
            nested_grid,
            res=level,
            aperture=7,
        )
        geom_representation = dggal_utils.get_grid_from_ids(
            dggs=dggs_instance,
            ids=uncompacted_ids,
        ).union_all()
    else:
        geom_representation = nested_grid.union_all()

    return geom_representation


def cover_geometry(dggs_name,
                   dggs_instance: str,
                   geometry,
                   levels: List[int],
                   levels_areas: Dict[int, float],
                   fix_thresholds: List[float],
                   min_threshold: float,
                   return_only_fix_grid: bool = False):

    # Initialize variables
    grid = gpd.GeoDataFrame()

    for level, threshold in zip(levels, fix_thresholds):
        if level == levels[0]:
            data: GeoDataFrame = __get_grid_from_geometry(
                dggs_name=dggs_name,
                dggs_instance=dggs_instance,
                geometry=geometry,
                level=level
            )

        else:
            # get children
            data: gpd.GeoDataFrame = __get_children(
                dggs_name=dggs_name,
                dggs_instance=dggs_instance,
                parents_ids=data_2_refine.cell_id.values,
                level=level
            )
                
        # overlay cells with geometry
        overlay: gpd.GeoDataFrame = get_overlay(
            data=data,
            cell_area=levels_areas[level],
            roi=geometry,
        )
        

        # Split data
        data_2_concat = overlay.loc[overlay.prop > threshold].copy()
        data_2_refine = overlay.loc[
            overlay.prop.between(min_threshold, threshold)].copy()
        
        
        # Updatate variables
        grid: pd.DataFrame = pd.concat([grid, data_2_concat],
                                    ignore_index=False)
        
        if (return_only_fix_grid) & (level == levels[-1]):
            return grid
        
        # Skip loop if there is no data 2 refine
        if data_2_refine.empty:
            break
        
    return grid, data_2_refine


def compute_experiment_metrics(
    fixed_grid: gpd.GeoDataFrame,
    var_grid: gpd.GeoDataFrame,
    finner_level: int,
    thresholds: List[float],
    geometry,
    dggs_instance,
    dggs_name: str,
):
    
    areas: List[float] = []
    ncells: List[int] = []
    int_areas: List[float] = []
    sm_dif_areas: List[float] = []
    
    for t in thresholds:
        partial_overlay = var_grid.loc[var_grid.prop > t]
        nested_grid = pd.concat([fixed_grid, partial_overlay],
                                            ignore_index=False)
        if nested_grid.empty:
            areas.append(0)
            int_areas.append(0)

            sm_dif_areas.append(shapely.transform(
                    geometry,
                    TRANSFORMER.transform,
                    interleaved=False
                ).area / AREA_SCALE_FACTOR)
            ncells.append(0)
        else:
            geom_representation = __get_geom_representation(
                nested_grid=nested_grid,
                dggs_name=dggs_name,
                dggs_instance=dggs_instance,
                level=finner_level,
            )

            areas.append(shapely.transform(
                geom_representation,
                TRANSFORMER.transform,
                interleaved=False
            ).area / AREA_SCALE_FACTOR)

            int_areas.append(shapely.transform(
                geometry.intersection(geom_representation),
                TRANSFORMER.transform,
                interleaved=False
            ).area / AREA_SCALE_FACTOR)

            sm_dif_areas.append(shapely.transform(
                geometry.symmetric_difference(geom_representation),
                TRANSFORMER.transform,
                interleaved=False
            ).area / AREA_SCALE_FACTOR)

            ncells.append(nested_grid.shape[0])

    return (areas, ncells,
            int_areas, sm_dif_areas)


def build_metrics_dataframe(
    polygons: gpd.GeoDataFrame,
    areas_data: List[float],
    ncells_data: List[float],
    int_areas_data: List[float],
    sm_dif_areas_data: List[float],
    col_names: List[str],
) -> pd.DataFrame:

    # Adding metrics to a DataFrame
    areas_data = pd.DataFrame(
        index=polygons.index.values,
        columns=[f'area_{col_name}' for col_name in col_names],
        data=areas_data,
    )

    ncells_data = pd.DataFrame(
        index=polygons.index.values,
        columns=[f'n_{col_name}' for col_name in col_names],
        data=ncells_data,
    )

    int_areas_data = pd.DataFrame(
        index=polygons.index.values,
        columns=[f'int_area_{col_name}' for col_name in col_names],
        data=int_areas_data,
    )

    sm_dif_areas_data = pd.DataFrame(
        index=polygons.index.values,
        columns=[f'sd_area_{col_name}' for col_name in col_names],
        data=sm_dif_areas_data,
    )

    experiment_data = areas_data.merge(
        ncells_data, right_index=True, left_index=True, how='inner')
    
    experiment_data = experiment_data.merge(
        int_areas_data, right_index=True, left_index=True, how='inner')

    experiment_data = experiment_data.merge(
        sm_dif_areas_data, right_index=True, left_index=True, how='inner')

    metrics_dataframe = polygons.merge(
        experiment_data,
        right_index=True,
        left_index=True,
        how='inner',
    )

    return metrics_dataframe


def run_experiment(
    dggs_name: str,
    dggs_instance,
    polygons: gpd.GeoDataFrame,
    levels: List[int],
    levels_areas: Dict[int, float],
    var_thresholds: List[float],
    fix_thresholds: List[float],
    min_threshold: float,
    col_names: List[str],
) -> pd.DataFrame:

    areas_data = []
    ncells_data = []
    int_areas_data = []
    sm_dif_areas_data = []

    
    for _, row in tqdm(polygons.iterrows(), total=polygons.shape[0]):
        
        # Initialize variables
        geometry = row.geometry

        # cover geometry with cells
        fixed_grid, var_grid = cover_geometry(
            dggs_name=dggs_name,
            dggs_instance=dggs_instance,
            geometry=geometry,
            levels=levels,
            levels_areas=levels_areas,
            fix_thresholds=fix_thresholds,
            min_threshold=min_threshold,
        )
    
        areas, ncells, int_areas, sm_dif_areas = compute_experiment_metrics(
            fixed_grid=fixed_grid,
            var_grid=var_grid,
            finner_level=levels[-1],
            thresholds=var_thresholds,
            geometry=geometry,
            dggs_name=dggs_name,
            dggs_instance=dggs_instance,
        )

        areas_data.append(areas)
        ncells_data.append(ncells)
        int_areas_data.append(int_areas)
        sm_dif_areas_data.append(sm_dif_areas)

    # Adding metrics to a DataFrame

    metrics_dataframe = build_metrics_dataframe(
        polygons=polygons,
        areas_data=areas_data,
        ncells_data=ncells_data,
        int_areas_data=int_areas_data,
        sm_dif_areas_data=sm_dif_areas_data,
        col_names=col_names,
    )

    return metrics_dataframe


def main(input_path: str, output_folder: str) -> NoReturn:
    polygons = gpd.read_parquet(input_path).to_crs(crs_definitions.WGS84)


    fill_strat_names = ['f2']
    hom_strat_names = ['h2']

    hom_strat = {
        'h1': [0.90, 0.80, 0.80],
        'h2': [0.95, 0.9, 0.90],
    }

    min_thresholds = {
        'h1': 0.1,
        'h2': 0.05,
    }

    dggs_levels={
        'IVEA7H': [10, 11, 12],
        'rHEALPix': [9, 10, 11],
    }

    DGGSs = [
        ('IVEA7H', dggal_utils.IVEA7H),
        # ('rHEALPix', rhealpix_utils.RDGGS3),
    ]

    for dggs_name, dggs_instance in DGGSs:
        max_levels = dggs_levels[dggs_name]
        if dggs_name == 'IVEA7H':
            levels_areas = {l: dggal_utils.get_level_cell_area(dggs=dggs_instance, level=l) for l in max_levels}
        else:
            levels_areas = {l: rhealpix_utils.get_level_cell_area(dggs=dggs_instance, level=l) for l in max_levels}

        fill_strat = {
            'f1': max_levels[:-1],
            'f2': max_levels,
        }

        for f, h in product(fill_strat_names, hom_strat_names):
            scenario = f'{f}{h}'

            print(f'DGGS: {dggs_name} \t Cenário: {scenario}')
            levels = fill_strat[f]
            min_threshold = min_thresholds[h]
            fix_thresholds = hom_strat[h]
            var_thresholds = [i / 10 for i in range(5, int(fix_thresholds[-1] * 10))]
            col_names = [f'0.{i}' for i in range(5, int(fix_thresholds[-1] * 10))]

            exp_data = run_experiment(
                dggs_name=dggs_name,
                dggs_instance=dggs_instance,
                polygons=polygons,
                levels=levels,
                levels_areas=levels_areas,
                var_thresholds=var_thresholds,
                fix_thresholds=fix_thresholds,
                min_threshold=min_threshold,
                col_names=col_names,
            )

            exp_data.to_parquet(
                f'{output_folder}/{dggs_name}_{scenario}.parquet'
            )


if __name__ == '__main__':
    input_path = f'{DATAPATH}/processed/sample/sample_with_metrics.parquet'
    output_folder = f'{DATAPATH}/processed/experiments'

    main(
        input_path=input_path,
        output_folder=output_folder
    )
