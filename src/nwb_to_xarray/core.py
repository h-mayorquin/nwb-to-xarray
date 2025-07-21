"""Core functions for converting all NWB data types to xarray."""

import pynwb
import xarray as xr
from typing import Any, Optional

from .ecephys import convert_nwb_ecephys_to_xarray
from .stimulus import convert_nwb_stimulus_to_xarray


def convert_nwb_to_xarray(nwbfile: pynwb.NWBFile, 
                         include_ecephys: bool = True,
                         include_ophys: bool = True,
                         include_stimulus: Optional[str] = None) -> dict[str, xr.Dataset]:
    """Convert NWB file to xarray datasets for all data types.
    
    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file object.
    include_ecephys : bool, default True
        Whether to convert electrophysiology data.
    include_ophys : bool, default True
        Whether to convert optical physiology data.
    include_stimulus : str, optional
        Name of stimulus intervals table to include (e.g., 'gabors_presentations').
        
    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary containing xarray datasets for each data type.
        Keys are data type names ('ecephys', 'ophys', 'stimulus').
    """
    datasets = {}
    
    # Convert electrophysiology data
    if include_ecephys:
        datasets['ecephys'] = convert_nwb_ecephys_to_xarray(nwbfile)
    
    # Convert optical physiology data
    if include_ophys:
        # For now, create empty dataset as ophys conversion is not implemented
        datasets['ophys'] = xr.Dataset(attrs={'ophys_note': 'Optical physiology conversion not yet implemented'})
    
    # Convert stimulus data
    if include_stimulus:
        datasets['stimulus'] = convert_nwb_stimulus_to_xarray(nwbfile, include_stimulus)
    
    return datasets