import matplotlib.pyplot as plt
import astropy.units as u
import sunpy.data.sample
import sunpy.map
import sunkit_image.radial as radial
import sunkit_image.enhance as enhance
from sunkit_image.utils import equally_spaced_bins

###########################################################################
# Load the sample AIA 171 image.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)


###########################################################################
# Implement the gamma function
import sunpy.map
import numpy as np
import matplotlib.pyplot as plt

def gamma_correct_map(smap, gamma=1.0):
    """
    Applies gamma correction to a SunPy map.

    Parameters:
        smap (sunpy.map.Map): The input SunPy map.
        gamma (float): Gamma correction factor (>1 brightens, <1 darkens).

    Returns:
        sunpy.map.Map: The gamma-corrected SunPy map.
    """
    # Normalize the data between 0 and 1
    data_min, data_max = smap.data.min(), smap.data.max()
    normalized_data = (smap.data - data_min) / (data_max - data_min)

    # Apply gamma correction
    gamma_corrected_data = normalized_data ** gamma

    # Rescale back to original range
    corrected_data = gamma_corrected_data * (data_max - data_min) + data_min

    # Create a new SunPy map with corrected data
    return sunpy.map.Map(corrected_data, smap.meta)

# Apply gamma correction
gamma_value = 0.65  # Adjust as needed
gamma_corrected_map = gamma_correct_map(aia_map, gamma=gamma_value)

# # Plot original and gamma-corrected maps
# fig, ax = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": aia_map.wcs})

# aia_map.plot(axes=ax[0])
# ax[0].set_title("Original Map")

# gamma_corrected_map.plot(axes=ax[1])
# ax[1].set_title(f"Gamma Corrected (γ={gamma_value})")

# plt.show(block=False)


###########################################################################
# Create radial bin edges and apply the NRGF, MSGN, and RHEF filters.

viggy = 1.5 * u.R_sun

radial_bin_edges = equally_spaced_bins(0, 2, aia_map.data.shape[0] // 1)
radial_bin_edges *= u.R_sun

base_nrgf = radial.nrgf(
    aia_map,
    radial_bin_edges=radial_bin_edges,
    application_radius=0.0 * u.R_sun,
    progress=True,
    # vignette=viggy,
)

import numpy as np
base_msgn = enhance.mgn(np.nan_to_num(aia_map, nan=0))

# order = 10
# attenuation_coefficients = radial.set_attenuation_coefficients(order)

# base_fnrgf = radial.fnrgf(
#     aia_map,
#     radial_bin_edges,
#     order,
#     attenuation_coefficients,
#     application_radius=1.0 * u.R_sun,
#     progress=True,
#     vignette=viggy,
# )

base_rhef = radial.rhef(
    aia_map,
    radial_bin_edges=radial_bin_edges,
    application_radius=0 * u.R_sun,
    progress=True,
    vignette=viggy,
    method="scipy",
)

###########################################################################
# Create subplots that share both x and y axes.

fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex="all", sharey="all", subplot_kw={"projection": aia_map})

###########################################################################
# Plot the original map and the filtered maps on the shared axes.
from matplotlib.colors import PowerNorm
gamma_corrected_map.plot(axes=axs[0, 0], clip_interval=(1, 99.99) * u.percent)
# aia_map.plot(axes=axs[0, 1], clip_interval=(1, 99.99) * u.percent)
# aia_map.plot(axes=axs[1, 0], clip_interval=(1, 99.99) * u.percent)
axs[0, 0].set_title("Gamma Corrected")

base_nrgf.plot(axes=axs[0, 1], clip_interval=(1, 99.5) * u.percent)
axs[0, 1].set_title("NRGF")

base_msgn.plot(axes=axs[1, 0], clip_interval=(10, 99.999) * u.percent)
axs[1, 0].set_title("MSGN")

base_rhef.plot(axes=axs[1, 1], clip_interval=(1, 99.9) * u.percent)
axs[1, 1].set_title("RHEF")

###########################################################################
# Set facecolor to black for all axes and hide tick labels for better visibility.

for ax in axs.flat:
    ax.set_facecolor("k")

axs[0, 0].coords[0].set_ticklabel_visible(False)
axs[0, 1].coords[0].set_ticklabel_visible(False)
axs[0, 1].coords[1].set_ticklabel_visible(False)
axs[1, 1].coords[1].set_ticklabel_visible(False)


fig.tight_layout()
# plt.show(block=True)
outfile = r"/Users/cgilbert/vscode/Sunback-Paper/Sunback-Paper-Expanded/Solar Image Processing and the Radial Histogram Equalizing Filter/fig/quadFirst2.pdf"
outfile = outfile.replace(".pdf", ".png")
plt.savefig(outfile, dpi=300)
plt.show()
