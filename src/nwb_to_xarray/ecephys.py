"""Electrophysiology data conversion utilities."""

import xarray as xr
import pynwb


def _convert_nwb_ecephys_to_xarray(nwbfile: pynwb.NWBFile) -> xr.Dataset:
    """Convert electrophysiology data from in-memory NWBFile to xarray Dataset.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        In-memory NWB file object.
        
    Returns
    -------
    xr.Dataset
        Electrophysiology data as xarray Dataset with dimensions:
        - time: timestamps (lazy index if starting_time and rate available)
        - channels: electrode indices
        
    Raises
    ------
    ValueError
        If no ElectricalSeries found in the file.
    """
    # Find all electrical series in acquisition using lambda filter
    is_electrical_series = lambda obj: hasattr(obj, 'data') and hasattr(obj, 'electrodes')
    electrical_series_list = list(filter(is_electrical_series, nwbfile.acquisition.values()))
    
    if not electrical_series_list:
        raise ValueError("No ElectricalSeries found in NWB file")
    
    data_vars = {}
    coords = {}
    attrs = {}
    
    # Get electrode information from nwbfile.electrodes
    electrode_table = nwbfile.electrodes
    electrode_table_indices = electrode_table.id[:]
    coords['electrode_table_index'] = electrode_table_indices
    
    # Add electrode table columns as DataArrays
    # Handle group_name explicitly
    group_names = [g.name for g in electrode_table["group"]]
    data_vars["group_name"] = xr.DataArray(
        data=group_names,
        dims=["electrode_table_index"],
        coords={"electrode_table_index": electrode_table_indices}
    )
    
    # Get all columns and filter out invalid ones
    invalid_columns = {'group', 'group_name'}
    valid_columns = [col for col in electrode_table.colnames if col not in invalid_columns]
    
    # Add other electrode table properties as DataArrays
    for col_name in valid_columns:
        data_vars[col_name] = xr.DataArray(
            data=electrode_table[col_name][:],
            dims=["electrode_table_index"],
            coords={"electrode_table_index": electrode_table_indices}
        )
    
    # Process each electrical series
    for electrical_series in electrical_series_list:
        series_name = electrical_series.name
        
        # Determine if has constant sampling rate
        has_constant_sampling_rate = hasattr(electrical_series, 'starting_time') and hasattr(electrical_series, 'rate')
        
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
            time_indices = xr.indexes.RangeIndex.arange(start=starting_time, stop=stop, step=step, dim="time")
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
                coords={'time': time_index, 'electrode_table_index': series_electrode_indices}
            )
        
        # Store electrical series data
        data_vars[f'{series_name}_data'] = data_array
        
        # Add series-specific attributes
        attrs[f'{series_name}_description'] = getattr(electrical_series, 'description', '')
        attrs[f'{series_name}_unit'] = getattr(electrical_series, 'unit', 'volts')
        attrs[f'{series_name}_conversion'] = getattr(electrical_series, 'conversion', 1.0)
        if hasattr(electrical_series, 'rate'):
            attrs[f'{series_name}_sampling_rate'] = electrical_series.rate
    
    # Create xarray Dataset
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    return dataset