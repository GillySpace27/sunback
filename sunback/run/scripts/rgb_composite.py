import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.visualization import make_lupton_rgb
from sunpy.map import Map
import numpy as np
from sunkit_image.enhance import mgn
from sunkit_image.radial import rhef
import os
import re


def load_image_files(directory):
    """Load FITS files from a directory and extract the wavelength from the filenames."""
    members = os.listdir(directory)
    fits = [m for m in members if m.endswith(".fits")]

    # Extract wavelength from filenames and load them into a dictionary
    image_files = {}
    for f in fits:
        match = re.findall(r"\d+", f)
        if match:
            wavelength = int(match[0])
            image_files[wavelength] = os.path.join(directory, f)
        else:
            print(f"Warning: No wavelength found in filename {f}")

    return image_files


def apply_filters(maps, filters):
    """Apply a sequence of filters to the maps."""
    filtered_maps = []
    for m in maps:
        for filt in filters:
            m = filt(
                m
            )  # Directly apply the filter to the map (filters should return a new Map)
        filtered_maps.append(m)
    return filtered_maps


def generate_composite(directory, wavelength_sets, operations):
    """Generate a composite image using FITS files from the specified directory."""
    # Load image files
    image_files = load_image_files(directory)

    # Ensure image files are correctly loaded for each wavelength set
    for wantwaves in wavelength_sets:
        for wv in wantwaves:
            if wv not in image_files:
                raise KeyError(
                    f"The requested wavelength {wv} is not found in the image files."
                )

    # Create figure with 2x2 layout (no projection specified here)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # Loop over the operations and wavelength sets to generate each plot
    for i, (filtname, filters) in enumerate(operations):
        row, col = divmod(i, 2)
        single = False

        # Get the appropriate wavelength maps
        wavelengths = wavelength_sets[i]
        if len(np.unique(wavelengths)) == 1:
            # Use a single map directly
            map_obj = Map(image_files[wavelengths[0]])  # Single map
            maps = [map_obj]  # Place into a list if needed for filtering
            single = True
        else:
            maps = [Map(image_files[wl]) for wl in wavelengths]  # Multiple maps

        # Apply the filters and handle nested structure
        maps_filtered = apply_filters(maps, filters)

        # Set the WCS projection for the current subplot
        ax = fig.add_subplot(2, 2, i + 1, projection=maps_filtered[0].wcs)

        if not single:
            # Create the RGB composite
            im_rgb = make_lupton_rgb(
                maps_filtered[0].data,
                maps_filtered[1].data,
                maps_filtered[2].data,
                Q=0,
                stretch=1,
            )
            ax.imshow(im_rgb)
        else:
            maps_filtered[0].plot(axes=ax)

        # Access the coordinates with WCSAxes
        lon, lat = ax.coords
        lon.set_axislabel("Helioprojective Longitude")
        lat.set_axislabel("Helioprojective Latitude")

        # Format the legend
        cmap = plt.cm.Set1
        custom_lines = [
            Line2D([0], [0], color=cmap(0), lw=4),
            Line2D([0], [0], color=cmap(2), lw=4),
            Line2D([0], [0], color=cmap(1), lw=4),
        ]
        legend_labels = [f"AIA {wl}" for wl in wavelengths]
        if ax is axs[0, 0]:
            ax.legend(custom_lines, legend_labels)
        ax.set_title(f"{filtname}")

    # Adjust layout
    fig.tight_layout()

    # Save the figure to the renders directory
    if not os.path.exists("renders"):
        os.makedirs("renders")

    file_name = "renders/composite_grid.pdf"
    plt.savefig(file_name, dpi=300)
    plt.show()


# Example usage with your directory
directory = "sunback_data/renders/background_server_lingon/rainbow/imgs/fits"
wavelength_sets = [
    [171, 193, 211],
    [171, 193, 211],
    [171, 193, 211],
    [171, 193, 211],
]

operations = [
    ("MGN", [mgn]),  # Ensure MGN returns valid data
    ("MGN", [mgn]),  # Ensure MGN returns valid data
    ("RHEF", [rhef]),  # RHEF only
    ("RHEF(MGN)", [mgn, rhef]),  # MGN then RHEF
]

generate_composite(directory, wavelength_sets, operations)
