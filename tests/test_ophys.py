"""Tests for ophys conversion functions."""

import pytest
from pynwb import read_nwb

from neuroconv.tools.testing.mock_interfaces import MockImagingInterface, MockSegmentationInterface
from neuroconv import ConverterPipe
from neuroconv.tools import configure_and_write_nwbfile

from nwb_to_xarray.ophys import _convert_nwb_ophys_to_xarray


def test_convert_mock_ophys_to_xarray(tmp_path):
    """Test conversion of mock ophys data to xarray Dataset."""
    num_rows = 10
    num_columns = 15
    
    imaging_interface = MockImagingInterface(num_rows=num_rows, num_columns=num_columns)
    segmentation_interface = MockSegmentationInterface(num_rows=num_rows, num_columns=num_columns)
    
    data_interfaces = {
        "imaging_interface": imaging_interface,
        "segmentation_interface": segmentation_interface,
    }
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces)
    
    nwbfile = converter_pipe.create_nwbfile()
    nwbfile_path = tmp_path / "test_ophys.nwb"
    
    configure_and_write_nwbfile(nwbfile, nwbfile_path=nwbfile_path)
    
    nwbfile = read_nwb(nwbfile_path)
    
    # Convert to xarray
    result = _convert_nwb_ophys_to_xarray(nwbfile)
    
    # Check for TwoPhotonSeries data
    assert "TwoPhotonSeries" in result.data_vars
    assert "time" in result.coords
    assert "y" in result.coords
    assert "x" in result.coords
    
    # Check for segmentation data
    assert "PlaneSegmentation_image_masks" in result.data_vars
    assert "roi" in result.coords
    
    # Check for fluorescence data
    fluorescence_vars = [var for var in result.data_vars if 'RoiResponseSeries' in var or 'Deconvolved' in var or 'Neuropil' in var]
    assert len(fluorescence_vars) > 0
    
    # Check TwoPhotonSeries shape
    two_photon_series = nwbfile.acquisition["TwoPhotonSeries"]
    expected_shape = two_photon_series.data.shape
    assert result["TwoPhotonSeries"].shape == expected_shape
    
    # Check time coordinate uses lazy indexing for starting_time + rate
    assert hasattr(two_photon_series, 'starting_time')
    assert hasattr(two_photon_series, 'rate')
    
    # Check attributes
    assert "TwoPhotonSeries_description" in result.attrs
    assert "TwoPhotonSeries_sampling_rate" in result.attrs


