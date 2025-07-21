# NWB to xarray

A Python library providing utilities to transform NWB (Neurodata Without Borders) data types to xarray format, enabling efficient analysis of neuroscience data using labeled multi-dimensional arrays.

## Purpose

This project aims to bridge the gap between the NWB data format and the xarray ecosystem, allowing neuroscientists to leverage the powerful features of xarray for their data analysis workflows. The library provides modular conversion functions for different NWB data types and showcases how these conversions can be used in real-world neuroscience analyses.

## Features

### 1. Lazy Loading of Time Series Data

The library implements efficient lazy loading for time series data using xarray's `RangeIndex`. When an NWB file contains time series with constant sampling rates (specified by `starting_time` and `rate`), we create lazy time coordinates that are computed on-demand rather than stored in memory:

```python
# Instead of creating a full time array in memory:
# time = np.arange(starting_time, starting_time + n_samples/rate, 1/rate)

# We use xarray's RangeIndex for lazy evaluation:
time_indices = xr.indexes.RangeIndex.arange(
    start=starting_time, stop=stop, step=step, dim="time"
)
```

This approach significantly reduces memory usage for large time series datasets.

### 2. Ragged Array Support for Variable-Length Data

Neuroscience data often contains variable-length arrays (e.g., spike times for different neurons, waveforms). The library uses the `ragged` package to handle these efficiently:

```python
# Convert variable-length spike times to CF-compliant format
spike_times_ragged = ragged.array(units["spike_times"][:])
content, counts = rio.to_cf_contiguous(spike_times_ragged)

# Store as separate arrays in xarray
dataset["unit_spike_times_content"] = content_array  # All spike times concatenated
dataset["unit_spike_times_counts"] = counts_array    # Number of spikes per unit
```

This allows xarray datasets to be serialized to formats like NetCDF while preserving the variable-length structure.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nwb-to-xarray.git
cd nwb-to-xarray

# Install with pip (recommended to use a virtual environment)
pip install -e .
```

## Usage

### Basic Conversion

```python
from pynwb import NWBHDF5IO
from nwb_to_xarray import convert_nwb_to_xarray

# Load NWB file
with NWBHDF5IO("path/to/file.nwb", "r") as io:
    nwbfile = io.read()

# Convert all data types
datasets = convert_nwb_to_xarray(
    nwbfile,
    include_ecephys=True,
    include_ophys=True,
    include_stimulus="gabors_presentations"
)

# Access individual datasets
ecephys_data = datasets['ecephys']
stimulus_data = datasets['stimulus']
```

### Module-Specific Conversions

```python
from nwb_to_xarray import convert_nwb_ecephys_to_xarray, convert_nwb_stimulus_to_xarray

# Convert only electrophysiology data
ecephys_dataset = convert_nwb_ecephys_to_xarray(nwbfile)

# Convert specific stimulus data
stimulus_dataset = convert_nwb_stimulus_to_xarray(nwbfile, "gabors_presentations")
```

### Working with Ragged Arrays

```python
from nwb_to_xarray import reconstruct_ragged_data

# Reconstruct spike times from CF format
spike_times = reconstruct_ragged_data(ecephys_dataset, 'spike_times')

# Access spike times for a specific unit
unit_0_spikes = spike_times[0]
```

## Example Analysis

See `scripts/receptive_fields_xarray.py` for a complete example of using this library to analyze receptive fields from visual neuroscience data. This script demonstrates:

- Loading NWB data and converting to xarray
- Selecting high-quality units based on quality metrics
- Computing receptive fields using spike responses to visual stimuli
- Visualizing results using matplotlib

