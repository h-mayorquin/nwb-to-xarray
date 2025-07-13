"""Tests for ecephys conversion functions."""

from pynwb import read_nwb

from neuroconv.tools.testing.mock_interfaces import MockRecordingInterface
from neuroconv import ConverterPipe
from neuroconv.tools import configure_and_write_nwbfile

from nwb_to_xarray.ecephys import _convert_nwb_ecephys_to_xarray


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
    result = _convert_nwb_ecephys_to_xarray(nwbfile)
    
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


