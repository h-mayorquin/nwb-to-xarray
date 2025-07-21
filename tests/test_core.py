"""Tests for core conversion functions."""

from datetime import datetime
from dateutil.tz import tzlocal
import pynwb
from pynwb import NWBFile

from neuroconv.tools.testing.mock_interfaces import MockRecordingInterface
from neuroconv import ConverterPipe
from neuroconv.tools import configure_and_write_nwbfile

from nwb_to_xarray.core import convert_nwb_to_xarray


def test_convert_nwb_to_xarray_all_types(tmp_path):
    """Test conversion of NWB file with all data types."""
    # Create mock ecephys data
    recording_interface = MockRecordingInterface()
    data_interfaces = {"recording_interface": recording_interface}
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces)
    nwbfile = converter_pipe.create_nwbfile()
    
    # Add stimulus data
    stim_table = nwbfile.create_time_intervals(
        name="test_stimulus",
        description="Test stimulus"
    )
    stim_table.add_column(name="x_position", description="X position")
    stim_table.add_row(start_time=0.0, stop_time=1.0, x_position=10.0)
    
    # Convert all types
    datasets = convert_nwb_to_xarray(
        nwbfile,
        include_ecephys=True,
        include_ophys=True,
        include_stimulus="test_stimulus"
    )
    
    # Check we got all requested types
    assert 'ecephys' in datasets
    assert 'ophys' in datasets
    assert 'stimulus' in datasets
    
    # Check ecephys has data
    assert len(datasets['ecephys'].data_vars) > 0
    
    # Check ophys has note about not being implemented
    assert 'ophys_note' in datasets['ophys'].attrs
    
    # Check stimulus has the data we added
    assert 'stim_x_position' in datasets['stimulus'].data_vars


def test_convert_nwb_to_xarray_selective():
    """Test selective conversion of NWB data types."""
    nwbfile = NWBFile(
        identifier="test",
        session_description="Test",
        session_start_time=datetime.now(tzlocal()),
    )
    
    # Convert only ecephys
    datasets = convert_nwb_to_xarray(
        nwbfile,
        include_ecephys=True,
        include_ophys=False,
        include_stimulus=None
    )
    
    # Check we only got ecephys
    assert 'ecephys' in datasets
    assert 'ophys' not in datasets
    assert 'stimulus' not in datasets


def test_convert_nwb_to_xarray_empty():
    """Test conversion of empty NWB file."""
    nwbfile = NWBFile(
        identifier="test",
        session_description="Test",
        session_start_time=datetime.now(tzlocal()),
    )
    
    # Convert empty file
    datasets = convert_nwb_to_xarray(
        nwbfile,
        include_ecephys=True,
        include_ophys=True,
        include_stimulus=None
    )
    
    # Should still return datasets, but they'll be empty
    assert 'ecephys' in datasets
    assert 'ophys' in datasets
    assert 'stimulus' not in datasets
    
    # Ecephys should be empty but valid
    assert len(datasets['ecephys'].data_vars) == 0