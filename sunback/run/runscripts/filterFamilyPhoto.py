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
# Create radial bin edges and apply the NRGF, MSGN, and RHEF filters.

viggy = 1.5 * u.R_sun

radial_bin_edges = equally_spaced_bins(0, 2, aia_map.data.shape[0] // 1)
radial_bin_edges *= u.R_sun

base_nrgf = radial.nrgf(
    aia_map,
    radial_bin_edges=radial_bin_edges,
    application_radius=0.0 * u.R_sun,
    progress=True,
    vignette=viggy,
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
aia_map.plot(axes=axs[0, 0], clip_interval=(1, 99.99) * u.percent)  #, norm=PowerNorm(gamma=2.2))

aia_map.plot(axes=axs[0, 1], clip_interval=(1, 99.99) * u.percent)
aia_map.plot(axes=axs[1, 0], clip_interval=(1, 99.99) * u.percent)
axs[0, 0].set_title("Original AIA 171")

base_nrgf.plot(axes=axs[0, 1], clip_interval=(1, 99.99) * u.percent)
axs[0, 1].set_title("NRGF")

base_msgn.plot(axes=axs[1, 0], clip_interval=(5, 99.99) * u.percent)
axs[1, 0].set_title("MSGN")

base_rhef.plot(axes=axs[1, 1], clip_interval=(1, 99.99) * u.percent)
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
plt.savefig(r"/Users/cgilbert/vscode/Sunback-Paper/Sunback-Paper-Expanded/Solar Image Processing and the Radial Histogram Equalizing Filter/fig/quadFirst2.pdf", dpi=300)
plt.show()
