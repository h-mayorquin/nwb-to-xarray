"""Optical physiology data conversion utilities."""

import xarray as xr
import numpy as np
import pynwb


def _convert_nwb_ophys_to_xarray(nwbfile: pynwb.NWBFile) -> xr.Dataset:
    """Convert optical physiology data from in-memory NWBFile to xarray Dataset.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        In-memory NWB file object.
        
    Returns
    -------
    xr.Dataset
        Optical physiology data as xarray Datasetwith dimensions:
        - time: timestamps (lazy index if starting_time and rate available)  
        - roi: ROI indices for fluorescence data
        - x, y: spatial dimensions for imaging data
        
    Raises
    ------
    ValueError
        If no optical physiology data found in the file.
    """
    data_vars = {}
    coords = {}
    attrs = {}
    
    # Find TwoPhoton imaging series using lambda filter
    is_two_photon_series = lambda obj: hasattr(obj, 'data') and 'TwoPhoton' in str(type(obj))
    two_photon_series_list = list(filter(is_two_photon_series, nwbfile.acquisition.values()))
    
    # Process TwoPhoton imaging series
    for imaging_series in two_photon_series_list:
        series_name = imaging_series.name
        
        # Determine if has constant sampling rate
        has_constant_sampling_rate = hasattr(imaging_series, 'starting_time') and hasattr(imaging_series, 'rate')
        
        # Get spatial dimensions
        y_dim = imaging_series.data.shape[1] if len(imaging_series.data.shape) > 1 else 1
        x_dim = imaging_series.data.shape[2] if len(imaging_series.data.shape) > 2 else 1
        
        # Set spatial coordinates
        if 'y' not in coords:
            coords['y'] = np.arange(y_dim)
        if 'x' not in coords:
            coords['x'] = np.arange(x_dim)
        
        # Create DataArray with time coordinate
        if has_constant_sampling_rate:
            # Create lazy time index using xarray RangeIndex
            starting_time = imaging_series.starting_time
            rate = imaging_series.rate
            n_samples = imaging_series.data.shape[0]
            step = 1.0 / rate
            stop = starting_time + n_samples * step
            time_indices = xr.indexes.RangeIndex.arange(start=starting_time, stop=stop, step=step, dim="time")
            time_coordinates = xr.Coordinates.from_xindex(time_indices)
            
            # Create DataArray and assign time coordinates
            if len(imaging_series.data.shape) == 3:  # time, y, x
                data_array = xr.DataArray(
                    data=imaging_series.data[:],
                    dims=['time', 'y', 'x'],
                    coords={'y': coords['y'], 'x': coords['x']}
                ).assign_coords(time_coordinates)
            elif len(imaging_series.data.shape) == 2:  # time, pixels
                pixels_coord = np.arange(imaging_series.data.shape[1])
                coords['pixels'] = pixels_coord
                data_array = xr.DataArray(
                    data=imaging_series.data[:],
                    dims=['time', 'pixels'],
                    coords={'pixels': pixels_coord}
                ).assign_coords(time_coordinates)
        else:
            # Use get_timestamps()
            time_index = imaging_series.get_timestamps()
            
            # Create DataArray with time vector
            if len(imaging_series.data.shape) == 3:  # time, y, x
                data_array = xr.DataArray(
                    data=imaging_series.data[:],
                    dims=['time', 'y', 'x'],
                    coords={'time': time_index, 'y': coords['y'], 'x': coords['x']}
                )
            elif len(imaging_series.data.shape) == 2:  # time, pixels
                pixels_coord = np.arange(imaging_series.data.shape[1])
                coords['pixels'] = pixels_coord
                data_array = xr.DataArray(
                    data=imaging_series.data[:],
                    dims=['time', 'pixels'],
                    coords={'time': time_index, 'pixels': pixels_coord}
                )
        
        # Store imaging data
        data_vars[f'{series_name}'] = data_array
        
        # Add imaging attributes
        attrs[f'{series_name}_description'] = getattr(imaging_series, 'description', '')
        attrs[f'{series_name}_unit'] = getattr(imaging_series, 'unit', '')
        if hasattr(imaging_series, 'rate'):
            attrs[f'{series_name}_sampling_rate'] = imaging_series.rate
    
    # Process optical physiology data
    if hasattr(nwbfile, 'processing') and 'ophys' in nwbfile.processing:
        ophys_module = nwbfile.processing['ophys']
        
        # Process fluorescence data
        if 'Fluorescence' in ophys_module.data_interfaces:
            fluorescence = ophys_module.data_interfaces['Fluorescence']
            
            for roi_response_series in fluorescence.roi_response_series.values():
                series_name = roi_response_series.name
                
                # Get ROI information
                if hasattr(roi_response_series, 'rois'):
                    roi_table = roi_response_series.rois.table
                    roi_ids = roi_table.id[:]
                    
                    # Set ROI coordinate if not already set
                    if 'roi' not in coords:
                        coords['roi'] = roi_ids
                    
                    # Add ROI table properties as DataArrays (similar to electrode table handling)
                    invalid_columns = {'id'}
                    valid_columns = [col for col in roi_table.colnames if col not in invalid_columns]
                    
                    for col_name in valid_columns:
                        if f'roi_{col_name}' not in data_vars:  # Avoid duplicates
                            col_data = roi_table[col_name][:]
                            # Handle different data shapes - some ROI table columns may be multi-dimensional
                            if col_data.ndim == 1:
                                data_vars[f'roi_{col_name}'] = xr.DataArray(
                                    data=col_data,
                                    dims=["roi"],
                                    coords={"roi": roi_ids}
                                )
                            elif col_data.ndim == 3:  # Likely image masks
                                data_vars[f'roi_{col_name}'] = xr.DataArray(
                                    data=col_data,
                                    dims=["roi", "y", "x"],
                                    coords={
                                        "roi": roi_ids,
                                        "y": np.arange(col_data.shape[1]),
                                        "x": np.arange(col_data.shape[2])
                                    }
                                )
                else:
                    # Fallback to simple indexing
                    roi_ids = np.arange(roi_response_series.data.shape[1])
                    if 'roi' not in coords:
                        coords['roi'] = roi_ids
                
                # Determine if has constant sampling rate
                has_constant_sampling_rate = hasattr(roi_response_series, 'starting_time') and hasattr(roi_response_series, 'rate')
                
                # Create DataArray with time coordinate
                if has_constant_sampling_rate:
                    # Create lazy time index using xarray RangeIndex
                    starting_time = roi_response_series.starting_time
                    rate = roi_response_series.rate
                    n_samples = roi_response_series.data.shape[0]
                    step = 1.0 / rate
                    stop = starting_time + n_samples * step
                    time_indices = xr.indexes.RangeIndex.arange(start=starting_time, stop=stop, step=step, dim="time")
                    time_coordinates = xr.Coordinates.from_xindex(time_indices)
                    
                    # Create DataArray and assign time coordinates
                    data_array = xr.DataArray(
                        data=roi_response_series.data[:],
                        dims=['time', 'roi'],
                        coords={'roi': roi_ids}
                    ).assign_coords(time_coordinates)
                else:
                    # Use get_timestamps()
                    time_index = roi_response_series.get_timestamps()
                    
                    # Create DataArray with time vector
                    data_array = xr.DataArray(
                        data=roi_response_series.data[:],
                        dims=['time', 'roi'],
                        coords={'time': time_index, 'roi': roi_ids}
                    )
                
                # Store fluorescence data
                data_vars[series_name] = data_array
                
                # Add series attributes
                attrs[f'{series_name}_description'] = getattr(roi_response_series, 'description', '')
                attrs[f'{series_name}_unit'] = getattr(roi_response_series, 'unit', '')
                if hasattr(roi_response_series, 'rate'):
                    attrs[f'{series_name}_sampling_rate'] = roi_response_series.rate
        
        # Process DfOverF data similarly
        if 'DfOverF' in ophys_module.data_interfaces:
            df_over_f = ophys_module.data_interfaces['DfOverF']
            
            for roi_response_series in df_over_f.roi_response_series.values():
                series_name = f"{roi_response_series.name}_df_over_f"
                
                # Get ROI information
                if hasattr(roi_response_series, 'rois'):
                    roi_ids = roi_response_series.rois.table.id[:]
                else:
                    roi_ids = np.arange(roi_response_series.data.shape[1])
                
                # Determine if has constant sampling rate
                has_constant_sampling_rate = hasattr(roi_response_series, 'starting_time') and hasattr(roi_response_series, 'rate')
                
                # Create DataArray with time coordinate
                if has_constant_sampling_rate:
                    # Create lazy time index using xarray RangeIndex
                    starting_time = roi_response_series.starting_time
                    rate = roi_response_series.rate
                    n_samples = roi_response_series.data.shape[0]
                    step = 1.0 / rate
                    stop = starting_time + n_samples * step
                    time_indices = xr.indexes.RangeIndex.arange(start=starting_time, stop=stop, step=step, dim="time")
                    time_coordinates = xr.Coordinates.from_xindex(time_indices)
                    
                    # Create DataArray and assign time coordinates
                    data_array = xr.DataArray(
                        data=roi_response_series.data[:],
                        dims=['time', 'roi'],
                        coords={'roi': roi_ids}
                    ).assign_coords(time_coordinates)
                else:
                    # Use get_timestamps()
                    time_index = roi_response_series.get_timestamps()
                    
                    # Create DataArray with time vector
                    data_array = xr.DataArray(
                        data=roi_response_series.data[:],
                        dims=['time', 'roi'],
                        coords={'time': time_index, 'roi': roi_ids}
                    )
                
                # Store DfOverF data
                data_vars[series_name] = data_array
                
                # Add series attributes
                attrs[f'{series_name}_description'] = getattr(roi_response_series, 'description', '')
                attrs[f'{series_name}_unit'] = getattr(roi_response_series, 'unit', '')
                if hasattr(roi_response_series, 'rate'):
                    attrs[f'{series_name}_sampling_rate'] = roi_response_series.rate
        
        # Process image segmentation data
        if 'ImageSegmentation' in ophys_module.data_interfaces:
            image_segmentation = ophys_module.data_interfaces['ImageSegmentation']
            
            for plane_segmentation in image_segmentation.plane_segmentations.values():
                plane_name = plane_segmentation.name
                
                # Get imaging plane information
                if hasattr(plane_segmentation, 'imaging_plane'):
                    imaging_plane = plane_segmentation.imaging_plane
                    attrs[f'{plane_name}_imaging_plane_description'] = getattr(imaging_plane, 'description', '')
                    
                    if hasattr(imaging_plane, 'grid_spacing'):
                        attrs[f'{plane_name}_grid_spacing'] = imaging_plane.grid_spacing
                    if hasattr(imaging_plane, 'origin_coords'):
                        attrs[f'{plane_name}_origin_coords'] = imaging_plane.origin_coords
                
                # Store ROI masks if available
                if 'image_mask' in plane_segmentation:
                    masks = plane_segmentation['image_mask'][:]
                    roi_ids = plane_segmentation.id[:]
                    
                    # Set coordinates if not already set
                    if 'roi' not in coords:
                        coords['roi'] = roi_ids
                    if 'y' not in coords:
                        coords['y'] = np.arange(masks.shape[1])
                    if 'x' not in coords:
                        coords['x'] = np.arange(masks.shape[2])
                    
                    # Create DataArray for image masks
                    data_vars[f'{plane_name}_image_masks'] = xr.DataArray(
                        data=masks,
                        dims=['roi', 'y', 'x'],
                        coords={'roi': roi_ids, 'y': coords['y'], 'x': coords['x']}
                    )
    
    if not data_vars:
        raise ValueError("No optical physiology data found in NWB file")
    
    # Create xarray Dataset
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    return dataset