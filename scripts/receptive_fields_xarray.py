#!/usr/bin/env python3
"""
Receptive fields analysis using xarray transformations.

This script reproduces the results from the openscope_databook receptive_fields.ipynb
notebook but using xarray-based data structures and operations instead of pandas/numpy.
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pynwb import read_nwb

# Import local functions
from src.nwb_to_xarray import convert_nwb_to_xarray, reconstruct_ragged_data


def convert_to_xarray_dataset(nwbfile):
    """Convert NWB data to xarray using modular functions."""
    # Use the new modular conversion function
    datasets = convert_nwb_to_xarray(
        nwbfile, 
        include_ecephys=True,
        include_ophys=False,
        include_stimulus="gabors_presentations"
    )
    
    # Combine ecephys and stimulus datasets
    ds = datasets['ecephys']
    if 'stimulus' in datasets:
        # Merge stimulus data into the main dataset
        ds = xr.merge([ds, datasets['stimulus']])
    
    # For receptive field calculation, we need spike times in a convenient format
    # Extract spike times from ragged format and store for quick access
    if 'unit_spike_times_content' in ds and 'unit_spike_times_counts' in ds:
        try:
            spike_times_ragged = reconstruct_ragged_data(ds, 'spike_times')
            spike_times_dict = {}
            for i, unit_id in enumerate(ds.unit_id.values):
                spike_times_dict[unit_id] = np.array(spike_times_ragged[i])
            ds.attrs['spike_times_dict'] = spike_times_dict
        except Exception as e:
            print(f"Could not reconstruct spike times: {e}")
    
    return ds


def get_unit_location_xarray(ds, unit_idx):
    """Get brain location for a unit using DataFrame and xarray operations."""
    units_df = ds.attrs['units_df']
    peak_channel_id = units_df.iloc[unit_idx]['peak_channel_id']
    
    # Check if electrode data is available in the dataset
    if 'id' in ds.data_vars:
        # Find the electrode with this channel ID
        electrode_mask = ds.id == peak_channel_id
        location = ds.location.where(electrode_mask, drop=True).values
        return location[0] if len(location) > 0 else ''
    else:
        return ''


def select_units_xarray(ds, nwbfile, max_units=50):
    """Select high-quality units using xarray operations and direct electrode access."""
    # Create boolean masks using xarray operations
    snr_mask = ds.unit_snr > 1
    isi_mask = ds.unit_isi_violations < 1
    firing_rate_mask = ds.unit_firing_rate > 0.1
    
    # Get brain locations directly from nwbfile.electrodes
    peak_channel_ids = ds.unit_peak_channel_id.values
    electrode_ids = nwbfile.electrodes.id[:]
    electrode_locations = nwbfile.electrodes["location"][:]
    
    # Create mapping from electrode ID to location
    id_to_location = dict(zip(electrode_ids, electrode_locations))
    
    # Get locations for all units
    locations = [id_to_location.get(peak_id, '') for peak_id in peak_channel_ids]
    
    # Create location mask for VISam using xarray
    location_mask = xr.DataArray(
        data=[loc == "VISam" for loc in locations],
        dims=["unit_id"],
        coords={"unit_id": ds.unit_id}
    )
    
    # Combine all masks using xarray operations
    quality_mask = snr_mask & isi_mask & firing_rate_mask & location_mask
    
    # Select units that meet criteria
    selected_unit_ids = ds.unit_id.where(quality_mask, drop=True).values
    
    # Limit number of units for faster processing
    if len(selected_unit_ids) > max_units:
        selected_unit_ids = selected_unit_ids[:max_units]
        print(f"Limited to first {max_units} units for faster processing")
    
    return selected_unit_ids


def get_stimulus_coordinates_xarray(ds):
    """Get unique x and y coordinates from stimulus table using xarray."""
    x_positions = np.sort(np.unique(ds.stim_x_position.values))
    y_positions = np.sort(np.unique(ds.stim_y_position.values))
    field_units = "deg"  # Default units for visual field positions
    
    return x_positions, y_positions, field_units


def compute_receptive_field_xarray(ds, unit_id, x_positions, y_positions):
    """Compute receptive field for a unit using xarray operations."""
    # Get spike times for this unit from stored dictionary
    spike_times_dict = ds.attrs['spike_times_dict']
    spike_times = spike_times_dict[unit_id]
    
    # Initialize receptive field array
    unit_rf = np.zeros([len(y_positions), len(x_positions)])
    
    # For each position in the receptive field
    for xi, x in enumerate(x_positions):
        for yi, y in enumerate(y_positions):
            # Find stimulus presentations at this position using xarray
            position_mask = (ds.stim_x_position == x) & (ds.stim_y_position == y)
            stim_times = ds.stim_start_time.where(position_mask, drop=True).values
            
            # Count spikes in response to stimuli at this position
            response_spike_count = 0
            for stim_time in stim_times:
                # Count spikes within 0.2 seconds after stimulus
                start_idx, end_idx = np.searchsorted(spike_times, [stim_time, stim_time + 0.2])
                response_spike_count += end_idx - start_idx
            
            unit_rf[yi, xi] = response_spike_count
    
    return unit_rf


def compute_all_receptive_fields_xarray(ds, selected_unit_ids, x_positions, y_positions):
    """Compute receptive fields for all selected units using xarray."""
    unit_rfs = []
    
    for unit_id in selected_unit_ids:
        rf = compute_receptive_field_xarray(ds, unit_id, x_positions, y_positions)
        unit_rfs.append(rf)
    
    # Convert to xarray DataArray
    rf_coords = {
        'unit_id': selected_unit_ids,
        'y_position': y_positions,
        'x_position': x_positions
    }
    
    rf_data = xr.DataArray(
        data=np.array(unit_rfs),
        dims=['unit_id', 'y_position', 'x_position'],
        coords=rf_coords,
        name='receptive_field_response'
    )
    
    return rf_data


def plot_receptive_fields_xarray(rf_data, x_positions, y_positions, field_units):
    """Plot receptive fields using xarray data with larger, cleaner visualization."""
    n_units = len(rf_data.unit_id)
    n_rows = n_units // 10 + (1 if n_units % 10 != 0 else 0)
    
    # Make figure larger with more space for imshow plots
    fig, axes = plt.subplots(n_rows, 10, figsize=(20, n_rows * 2.5))
    
    # Handle case where there's <= 10 rfs
    if n_units <= 10:
        axes = axes.reshape((1, -1)) if n_rows == 1 else axes
    elif len(axes.shape) == 1:
        axes = axes.reshape((1, axes.shape[0]))
    
    # Plot each receptive field
    for i, unit_id in enumerate(rf_data.unit_id):
        if i >= n_rows * 10:
            break
            
        ax_row = i // 10
        ax_col = i % 10
        
        # Get receptive field for this unit
        rf = rf_data.sel(unit_id=unit_id).values
        
        # Plot with larger imshow and cleaner appearance
        im = axes[ax_row][ax_col].imshow(rf, origin="lower", aspect='equal')
        axes[ax_row][ax_col].set_title(f'Unit {int(unit_id)}', fontsize=10, pad=10)
        
        # Remove tick marks and labels for cleaner look
        axes[ax_row][ax_col].set_xticks([])
        axes[ax_row][ax_col].set_yticks([])
        
        # Add colorbar for the first plot to show scale
        if i == 0:
            plt.colorbar(im, ax=axes[ax_row][ax_col], fraction=0.046, pad=0.04)
    
    # Turn off axes for unused subplots
    for ax in axes.flat[n_units:]:
        ax.axis('off')
    
    # Add coordinate labels as figure text instead of individual plot labels
    fig.suptitle(f'Receptive Fields (coordinates in {field_units})\n' + 
                f'X: {x_positions[0]} to {x_positions[-1]}, Y: {y_positions[0]} to {y_positions[-1]}', 
                fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.2)
    plt.savefig("receptive_fields_xarray.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def main():
    """Main function to reproduce receptive fields analysis with xarray."""
    print("Loading NWB file...")
    # Use local downloaded file
    nwbfile = read_nwb("sub-716813540_ses-739448407.nwb")
    
    print("Converting to xarray dataset...")
    ds = convert_to_xarray_dataset(nwbfile)
    
    print("Selecting high-quality units from VISam...")
    selected_unit_ids = select_units_xarray(ds, nwbfile)
    print(f"Selected {len(selected_unit_ids)} units")
    
    if len(selected_unit_ids) == 0:
        raise ValueError("No units selected. Adjust selection criteria.")
    
    print("Getting stimulus coordinates...")
    x_positions, y_positions, field_units = get_stimulus_coordinates_xarray(ds)
    print(f"Stimulus positions: x={x_positions}, y={y_positions}, units={field_units}")
    
    print("Computing receptive fields...")
    rf_data = compute_all_receptive_fields_xarray(ds, selected_unit_ids, x_positions, y_positions)
    
    print("Plotting receptive fields...")
    fig = plot_receptive_fields_xarray(rf_data, x_positions, y_positions, field_units)
    
    # Save only the high-resolution figure
    print("Figure saved to receptive_fields_xarray.png")
    
    print("Analysis complete!")
    return ds, rf_data, fig


if __name__ == "__main__":
    ds, rf_data, fig = main()