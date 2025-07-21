"""Stimulus data conversion utilities."""

import numpy as np
import xarray as xr
import pynwb
from typing import Any


def stimulus_to_xarray(nwbfile: pynwb.NWBFile, stimulus_name: str) -> dict[str, Any]:
    """Convert NWB stimulus intervals to xarray DataArrays.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
    stimulus_name : str
        Name of the stimulus intervals table (e.g., 'gabors_presentations').
        
    Returns
    -------
    dict[str, Any]
        Dictionary containing data_vars, coords, and attrs for stimulus.
    """
    data_vars = {}
    coords = {}
    attrs = {}
    
    if stimulus_name not in nwbfile.intervals:
        attrs[f'{stimulus_name}_error'] = f'Stimulus {stimulus_name} not found in intervals'
        return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}
    
    stim_table = nwbfile.intervals[stimulus_name]
    stim_df = stim_table.to_dataframe()
    stim_ids = stim_df.index.values
    coords['stimulus_id'] = stim_ids
    
    # Serializable stimulus columns (numeric data)
    serializable_columns = [
        'start_time', 'stop_time', 'x_position', 'y_position', 
        'temporal_frequency', 'orientation', 'spatial_frequency', 
        'contrast', 'stimulus_block', 'opacity', 'stimulus_index'
    ]
    
    for col_name in serializable_columns:
        if col_name in stim_df.columns:
            data_vars[f'stim_{col_name}'] = xr.DataArray(
                data=stim_df[col_name].values,
                dims=["stimulus_id"],
                coords={"stimulus_id": stim_ids}
            )
    
    # String columns (convert to simple strings)
    string_columns = ['stimulus_name', 'mask', 'units', 'color', 'phase', 'size']
    
    for col_name in string_columns:
        if col_name in stim_df.columns:
            # Convert to string representation
            string_data = [str(val) for val in stim_df[col_name].values]
            data_vars[f'stim_{col_name}'] = xr.DataArray(
                data=string_data,
                dims=["stimulus_id"],
                coords={"stimulus_id": stim_ids}
            )
    
    # Tags column (extract string content from arrays)
    if 'tags' in stim_df.columns:
        tags_strings = []
        for tag_array in stim_df['tags'].values:
            if hasattr(tag_array, '__len__') and len(tag_array) > 0:
                tag_str = ','.join(str(tag) for tag in tag_array)
            else:
                tag_str = ''
            tags_strings.append(tag_str)
        
        data_vars['stim_tags'] = xr.DataArray(
            data=tags_strings,
            dims=["stimulus_id"],
            coords={"stimulus_id": stim_ids}
        )
    
    # Skip timeseries column (too complex for serialization)
    if 'timeseries' in stim_df.columns:
        attrs['stim_timeseries_note'] = 'TimeSeriesReference objects not serialized'
    
    return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}


def convert_nwb_stimulus_to_xarray(nwbfile: pynwb.NWBFile, stimulus_name: str) -> xr.Dataset:
    """Convert NWB stimulus data to xarray Dataset.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
    stimulus_name : str
        Name of the stimulus intervals table.
        
    Returns
    -------
    xr.Dataset
        Stimulus data as xarray Dataset.
    """
    result = stimulus_to_xarray(nwbfile, stimulus_name)
    
    dataset = xr.Dataset(
        data_vars=result['data_vars'],
        coords=result['coords'],
        attrs=result['attrs']
    )
    
    return dataset