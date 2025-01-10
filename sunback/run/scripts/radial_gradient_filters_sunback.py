import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.visualization import make_lupton_rgb
import sunpy.data.sample
from sunpy.map import Map
import numpy as np
import astropy.units as u
from sunkit_image.enhance import mgn
from sunkit_image.radial import rhef, nrgf
from functools import partial
import os


# Define clipping function
def clip_map(aia_map, lower_percent=1, upper_percent=99.99):
    data = aia_map.data
    lower_bound = np.nanpercentile(data, lower_percent)
    upper_bound = np.nanpercentile(data, upper_percent)
    clipped_data = np.clip(data, lower_bound, upper_bound)
    return aia_map.__class__(clipped_data, aia_map.meta)


# Normalize function
def normalize(I):
    I_min = np.nanmin(I)
    I_max = np.nanmax(I)
    return (I - I_min) / (I_max - I_min)


# Set operations and wavelengths for each of the 4 plots
operations = [
    ("RHEF(1p5)", [rhef]),  # RHEF only
    ("RHEF(NRGF)", [partial(nrgf, fill=np.nan), rhef]),  # NRGF then RHEF
    ("RHEF(MGN)", [mgn, rhef]),  # MGN then RHEF
    ("MGN(1p5)", [mgn]),  # MGN only
]

# Define wavelengths for each plot
wavelength_sets = [
    [171, 193, 211],  # Set for the first plot
    [171, 193, 211],  # Set for the second plot
    [171, 193, 211],  # Set for the third plot
    [171, 193, 211],  # Set for the fourth plot
]

# Define the sample image files for the wavelengths
image_files = {
    304: sunpy.data.sample.AIA_304_IMAGE,
    1600: sunpy.data.sample.AIA_1600_IMAGE,
    94: sunpy.data.sample.AIA_094_IMAGE,
    171: sunpy.data.sample.AIA_171_IMAGE,
    193: sunpy.data.sample.AIA_193_IMAGE,
    211: sunpy.data.sample.AIA_211_IMAGE,
    335: sunpy.data.sample.AIA_335_IMAGE,
}

# Create figure with 2x2 layout and shared x and y axes
fig, axs = plt.subplots(
    2,
    2,
    figsize=(8, 8),
    subplot_kw={"projection": Map(image_files[171]).wcs},
    sharex=True,
    sharey=True,
)

# Loop over the operations and wavelength sets to generate each plot
for i, (filtname, filters) in enumerate(operations):
    row, col = divmod(i, 2)

    # Get the appropriate wavelength maps
    wavelengths = wavelength_sets[i]
    maps = Map([image_files[wl] for wl in wavelengths])

    # Apply the filters in series (one after the other)
    maps_filtered = maps
    for filt in filters:
        maps_filtered = [filt(m) for m in maps_filtered]

    # Create the RGB composite
    im_rgb = make_lupton_rgb(
        maps_filtered[0].data,
        maps_filtered[1].data,
        maps_filtered[2].data,
        Q=0,
        stretch=1,
    )

    # Plot the composite
    ax = axs[row, col]
    im = ax.imshow(im_rgb)
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
    if ax is axs[0, 1]:
        legend = ax.legend(
            custom_lines, legend_labels, fontsize=9, frameon=False, loc="center"
        )
        plt.setp(legend.get_texts(), color="white")

    ax.set_title(f"{filtname}")

# Add supertitle
supertitle = f"AIA RGB Composite ({', '.join(map(str, wavelengths))} Angstroms)"
fig.suptitle(supertitle)

# Adjust layout and make axes shared
fig.tight_layout()

# Save the figure to the renders directory
if not os.path.exists("renders"):
    os.makedirs("renders")

file_name = "renders/composite_grid_1.pdf"
print(f"Saving to {os.path.abspath(file_name)}")
plt.savefig(file_name, dpi=400)
plt.show()
