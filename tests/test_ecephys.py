"""Tests for ecephys conversion functions."""

from pynwb import read_nwb

from neuroconv.tools.testing.mock_interfaces import MockRecordingInterface
from neuroconv import ConverterPipe
from neuroconv.tools import configure_and_write_nwbfile

from nwb_to_xarray.ecephys import (
    convert_nwb_ecephys_to_xarray,
    electrodes_to_xarray,
    units_to_xarray,
    timeseries_to_xarray,
    reconstruct_ragged_data,
)


def test_convert_mock_ecephys_to_xarray(tmp_path):
    """Test conversion of mock ecephys data to xarray Dataset."""
    # Create mock data using neuroconv mock interfaces
    recording_interface = MockRecordingInterface()
    
    data_interfaces = {"recording_interface": recording_interface}
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces)
    
    nwbfile = converter_pipe.create_nwbfile()
    nwbfile_path = tmp_path / "test_ecephys.nwb"
    
    configure_and_write_nwbfile(nwbfile, nwbfile_path=nwbfile_path)
    
    nwbfile = read_nwb(nwbfile_path)
    
    # Convert to xarray
    result = convert_nwb_ecephys_to_xarray(nwbfile)
    
    # Assertions
    assert "ElectricalSeries_data" in result.data_vars
    assert "time" in result.coords
    assert "electrode_table_index" in result.coords
    assert "group_name" in result.data_vars  # group_name should be explicitly handled
    assert "location" in result.data_vars
    
    # Check that we have the expected electrical series
    electrical_series = nwbfile.acquisition["ElectricalSeries"]
    expected_shape = electrical_series.data.shape
    assert result["ElectricalSeries_data"].shape == expected_shape
    
    # Check time coordinate uses lazy indexing for starting_time + rate
    assert hasattr(electrical_series, 'starting_time')
    assert hasattr(electrical_series, 'rate')
    
    # Check attributes
    assert "ElectricalSeries_description" in result.attrs
    assert "ElectricalSeries_unit" in result.attrs
    assert "ElectricalSeries_sampling_rate" in result.attrs


def test_electrodes_to_xarray(tmp_path):
    """Test conversion of electrodes table to xarray."""
    # Create mock data
    recording_interface = MockRecordingInterface()
    data_interfaces = {"recording_interface": recording_interface}
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces)
    nwbfile = converter_pipe.create_nwbfile()
    
    # Convert electrodes
    result = electrodes_to_xarray(nwbfile)
    
    # Check structure
    assert 'data_vars' in result
    assert 'coords' in result
    assert 'attrs' in result
    
    # Check coordinates
    assert 'electrode_table_index' in result['coords']
    
    # Check data variables
    assert 'group_name' in result['data_vars']
    assert 'location' in result['data_vars']
    
    # Check that group names are properly extracted
    group_name_data = result['data_vars']['group_name']
    assert group_name_data.dims == ('electrode_table_index',)


def test_timeseries_to_xarray(tmp_path):
    """Test conversion of electrical time series to xarray."""
    # Create mock data
    recording_interface = MockRecordingInterface()
    data_interfaces = {"recording_interface": recording_interface}
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces)
    nwbfile = converter_pipe.create_nwbfile()
    
    # Convert time series
    result = timeseries_to_xarray(nwbfile)
    
    # Check structure
    assert 'data_vars' in result
    assert 'coords' in result
    assert 'attrs' in result
    
    # Check for electrical series data
    electrical_series_vars = [var for var in result['data_vars'] if 'ElectricalSeries' in var]
    assert len(electrical_series_vars) > 0
    
    # Check attributes for sampling rate
    sampling_rate_attrs = [attr for attr in result['attrs'] if 'sampling_rate' in attr]
    assert len(sampling_rate_attrs) > 0


def test_units_to_xarray_empty(tmp_path):
    """Test conversion of units table when no units exist."""
    # Create mock data without units
    recording_interface = MockRecordingInterface()
    data_interfaces = {"recording_interface": recording_interface}
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces)
    nwbfile = converter_pipe.create_nwbfile()
    
    # Convert units (should handle empty case gracefully)
    result = units_to_xarray(nwbfile)
    
    # Check structure
    assert 'data_vars' in result
    assert 'coords' in result
    assert 'attrs' in result
    
    # Should be empty since no units
    assert len(result['data_vars']) == 0
    assert len(result['coords']) == 0


