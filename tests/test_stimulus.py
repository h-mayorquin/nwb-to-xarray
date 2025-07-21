"""Tests for stimulus conversion functions."""

import numpy as np
import pandas as pd
import pynwb
from pynwb import NWBFile, TimeSeries
from datetime import datetime
from dateutil.tz import tzlocal

from nwb_to_xarray.stimulus import (
    stimulus_to_xarray,
    convert_nwb_stimulus_to_xarray,
)


def create_test_nwbfile_with_stimulus():
    """Create a test NWB file with stimulus intervals."""
    nwbfile = NWBFile(
        identifier="test_stimulus",
        session_description="Test stimulus conversion",
        session_start_time=datetime.now(tzlocal()),
    )
    
    # Create stimulus intervals table
    stim_table = nwbfile.create_time_intervals(
        name="gabors_presentations",
        description="Gabor stimulus presentations"
    )
    
    # Add columns
    stim_table.add_column(name="x_position", description="X position")
    stim_table.add_column(name="y_position", description="Y position")
    stim_table.add_column(name="orientation", description="Orientation")
    stim_table.add_column(name="spatial_frequency", description="Spatial frequency")
    stim_table.add_column(name="contrast", description="Contrast")
    stim_table.add_column(name="stimulus_name", description="Stimulus name")
    stim_table.add_column(name="tags", description="Tags", index=True)
    
    # Add some test data
    for i in range(10):
        stim_table.add_row(
            start_time=float(i),
            stop_time=float(i + 0.5),
            x_position=float(i * 10),
            y_position=float(i * 5),
            orientation=float(i * 30),
            spatial_frequency=0.04,
            contrast=0.8,
            stimulus_name=f"gabor_{i}",
            tags=["test", f"trial_{i}"]
        )
    
    return nwbfile


def test_stimulus_to_xarray():
    """Test conversion of stimulus intervals to xarray."""
    nwbfile = create_test_nwbfile_with_stimulus()
    
    # Convert stimulus
    result = stimulus_to_xarray(nwbfile, "gabors_presentations")
    
    # Check structure
    assert 'data_vars' in result
    assert 'coords' in result
    assert 'attrs' in result
    
    # Check coordinates
    assert 'stimulus_id' in result['coords']
    assert len(result['coords']['stimulus_id']) == 10
    
    # Check data variables
    assert 'stim_start_time' in result['data_vars']
    assert 'stim_stop_time' in result['data_vars']
    assert 'stim_x_position' in result['data_vars']
    assert 'stim_y_position' in result['data_vars']
    assert 'stim_orientation' in result['data_vars']
    assert 'stim_spatial_frequency' in result['data_vars']
    assert 'stim_contrast' in result['data_vars']
    assert 'stim_stimulus_name' in result['data_vars']
    assert 'stim_tags' in result['data_vars']
    
    # Check tags conversion
    tags_data = result['data_vars']['stim_tags']
    assert tags_data.dims == ('stimulus_id',)
    assert isinstance(tags_data.values[0], str)
    assert 'test,trial_0' in tags_data.values[0]


def test_stimulus_to_xarray_missing():
    """Test conversion when stimulus table doesn't exist."""
    nwbfile = NWBFile(
        identifier="test",
        session_description="Test",
        session_start_time=datetime.now(tzlocal()),
    )
    
    # Convert non-existent stimulus
    result = stimulus_to_xarray(nwbfile, "nonexistent_stimulus")
    
    # Should return empty structure with error note
    assert 'data_vars' in result
    assert 'coords' in result
    assert 'attrs' in result
    assert len(result['data_vars']) == 0
    assert 'nonexistent_stimulus_error' in result['attrs']


def test_convert_nwb_stimulus_to_xarray():
    """Test high-level stimulus conversion function."""
    nwbfile = create_test_nwbfile_with_stimulus()
    
    # Convert to dataset
    ds = convert_nwb_stimulus_to_xarray(nwbfile, "gabors_presentations")
    
    # Check it's an xarray Dataset
    import xarray as xr
    assert isinstance(ds, xr.Dataset)
    
    # Check data variables
    assert 'stim_x_position' in ds.data_vars
    assert 'stim_y_position' in ds.data_vars
    
    # Check coordinates
    assert 'stimulus_id' in ds.coords