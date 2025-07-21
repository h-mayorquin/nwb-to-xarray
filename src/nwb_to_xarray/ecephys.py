"""Electrophysiology data conversion utilities."""

import numpy as np
import xarray as xr
import pynwb
import ragged
import ragged.io as rio
from typing import Any, Optional


def electrodes_to_xarray(nwbfile: pynwb.NWBFile) -> dict[str, Any]:
    """Convert NWB electrodes table to xarray DataArrays.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing data_vars, coords, and attrs for electrodes.
    """
    data_vars = {}
    coords = {}
    attrs = {}
    
    if nwbfile.electrodes is None:
        return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}
    
    electrode_table = nwbfile.electrodes
    electrode_ids = electrode_table.id[:]
    coords['electrode_table_index'] = electrode_ids
    
    # Handle group names specially
    if 'group' in electrode_table.colnames:
        group_names = [g.name for g in electrode_table["group"]]
        data_vars["group_name"] = xr.DataArray(
            data=group_names,
            dims=["electrode_table_index"],
            coords={"electrode_table_index": electrode_ids}
        )
    
    # Add other electrode properties
    invalid_columns = {'group', 'group_name'}
    valid_columns = [col for col in electrode_table.colnames if col not in invalid_columns]
    
    for col_name in valid_columns:
        data_vars[col_name] = xr.DataArray(
            data=electrode_table[col_name][:],
            dims=["electrode_table_index"],
            coords={"electrode_table_index": electrode_ids}
        )
    
    return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}


def units_to_xarray(nwbfile: pynwb.NWBFile) -> dict[str, Any]:
    """Convert NWB units table to xarray DataArrays with ragged array support.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing data_vars, coords, and attrs for units.
    """
    data_vars = {}
    coords = {}
    attrs = {}
    
    if nwbfile.units is None:
        return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}
    
    units = nwbfile.units
    unit_ids = units.id[:]
    coords['unit_id'] = unit_ids
    
    # Scalar columns (regular arrays)
    scalar_columns = [
        'peak_channel_id', 'snr', 'isi_violations', 'firing_rate', 
        'quality', 'amplitude', 'waveform_halfwidth', 'velocity_below',
        'l_ratio', 'amplitude_cutoff', 'spread', 'nn_miss_rate',
        'isolation_distance', 'silhouette_score', 'velocity_above',
        'cumulative_drift', 'waveform_duration', 'recovery_slope'
    ]
    
    for col_name in scalar_columns:
        if col_name in units.colnames:
            data_vars[f'unit_{col_name}'] = xr.DataArray(
                data=units[col_name][:],
                dims=["unit_id"],
                coords={"unit_id": unit_ids}
            )
    
    # Ragged columns (variable-length arrays)
    ragged_columns = ['spike_times', 'spike_amplitudes', 'waveform_mean']
    
    for col_name in ragged_columns:
        if col_name in units.colnames:
            # Convert to ragged array
            ragged_data = ragged.array(units[col_name][:])
            
            # Convert to CF format for serialization
            content, counts = rio.to_cf_contiguous(ragged_data)
            
            # Add as separate xarray variables
            data_vars[f'unit_{col_name}_content'] = xr.DataArray(
                data=np.array(content, dtype=np.float64),
                dims=[f'{col_name}_index'],
                name=f'{col_name}_content'
            )
            
            data_vars[f'unit_{col_name}_counts'] = xr.DataArray(
                data=np.array(counts, dtype=np.int64),
                dims=["unit_id"],
                coords={"unit_id": unit_ids},
                name=f'{col_name}_counts'
            )
            
            # Store reconstruction info
            attrs[f'{col_name}_ragged_format'] = 'cf_contiguous'
    
    return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}


def timeseries_to_xarray(nwbfile: pynwb.NWBFile) -> dict[str, Any]:
    """Convert NWB electrical time series to xarray DataArrays.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing data_vars, coords, and attrs for time series.
    """
    data_vars = {}
    coords = {}
    attrs = {}
    
    # Find all electrical series in acquisition
    is_electrical_series = lambda obj: hasattr(obj, 'data') and hasattr(obj, 'electrodes')
    electrical_series_list = list(filter(is_electrical_series, nwbfile.acquisition.values()))
    
    if not electrical_series_list:
        attrs['timeseries_note'] = 'No ElectricalSeries found in NWB file'
        return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}
    
    # Process each electrical series
    for electrical_series in electrical_series_list:
        series_name = electrical_series.name
        
        # Determine if has constant sampling rate
        has_constant_sampling_rate = (
            hasattr(electrical_series, 'starting_time') and 
            hasattr(electrical_series, 'rate')
        )
        
        # Get electrode indices for this series
        series_electrode_indices = electrical_series.electrodes.data[:]
        
        # Create DataArray with time coordinate
        if has_constant_sampling_rate:
            # Create lazy time index using xarray RangeIndex
            starting_time = electrical_series.starting_time
            rate = electrical_series.rate
            n_samples = electrical_series.data.shape[0]
            step = 1.0 / rate
            stop = starting_time + n_samples * step
            time_indices = xr.indexes.RangeIndex.arange(
                start=starting_time, stop=stop, step=step, dim="time"
            )
            time_coordinates = xr.Coordinates.from_xindex(time_indices)
            
            # Create DataArray and assign time coordinates
            data_array = xr.DataArray(
                data=electrical_series.data[:],
                dims=['time', 'electrode_table_index'],
                coords={'electrode_table_index': series_electrode_indices}
            ).assign_coords(time_coordinates)
        else:
            # Use get_timestamps()
            time_index = electrical_series.get_timestamps()
            
            # Create DataArray with time vector
            data_array = xr.DataArray(
                data=electrical_series.data[:],
                dims=['time', 'electrode_table_index'],
                coords={
                    'time': time_index, 
                    'electrode_table_index': series_electrode_indices
                }
            )
        
        # Store electrical series data
        data_vars[f'{series_name}_data'] = data_array
        
        # Add series-specific attributes
        attrs[f'{series_name}_description'] = getattr(electrical_series, 'description', '')
        attrs[f'{series_name}_unit'] = getattr(electrical_series, 'unit', 'volts')
        attrs[f'{series_name}_conversion'] = getattr(electrical_series, 'conversion', 1.0)
        if hasattr(electrical_series, 'rate'):
            attrs[f'{series_name}_sampling_rate'] = electrical_series.rate
    
    return {'data_vars': data_vars, 'coords': coords, 'attrs': attrs}




def reconstruct_ragged_data(ds: xr.Dataset, col_name: str) -> 'ragged.array':
    """Reconstruct ragged array from CF format in xarray dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing CF format ragged data.
    col_name : str
        Name of the ragged column (e.g., 'spike_times').
        
    Returns
    -------
    ragged.array
        Reconstructed ragged array.
    """
    content_key = f'unit_{col_name}_content'
    counts_key = f'unit_{col_name}_counts'
    
    if content_key not in ds or counts_key not in ds:
        raise ValueError(f"No ragged data found for {col_name}")
    
    # Convert back to ragged arrays
    content = ragged.array(ds[content_key].values)
    counts = ragged.array(ds[counts_key].values)
    
    # Reconstruct original ragged structure
    return rio.from_cf_contiguous(content, counts)


def convert_nwb_ecephys_to_xarray(nwbfile: pynwb.NWBFile) -> xr.Dataset:
    """Convert NWB electrophysiology data to xarray Dataset.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
        
    Returns
    -------
    xr.Dataset
        Electrophysiology data as xarray Dataset.
    """
    all_data_vars = {}
    all_coords = {}
    all_attrs = {}
    
    # Convert electrodes
    electrodes_result = electrodes_to_xarray(nwbfile)
    all_data_vars.update(electrodes_result['data_vars'])
    all_coords.update(electrodes_result['coords'])
    all_attrs.update(electrodes_result['attrs'])
    
    # Convert units
    units_result = units_to_xarray(nwbfile)
    all_data_vars.update(units_result['data_vars'])
    all_coords.update(units_result['coords'])
    all_attrs.update(units_result['attrs'])
    
    # Convert time series if available
    timeseries_result = timeseries_to_xarray(nwbfile)
    all_data_vars.update(timeseries_result['data_vars'])
    all_coords.update(timeseries_result['coords'])
    all_attrs.update(timeseries_result['attrs'])
    
    # Create the combined dataset
    dataset = xr.Dataset(
        data_vars=all_data_vars,
        coords=all_coords,
        attrs=all_attrs
    )
    
    return dataset


def _convert_nwb_ecephys_to_xarray(nwbfile: pynwb.NWBFile) -> xr.Dataset:
    """Legacy function for backwards compatibility."""
    return convert_nwb_ecephys_to_xarray(nwbfile)