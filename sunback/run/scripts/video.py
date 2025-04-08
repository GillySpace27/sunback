import sunpy.map
from sunpy.net import Fido, attrs as a
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import astropy.units as u
from sunkit_image.radial import rhef
from tqdm import tqdm

# Define query parameters
start_time = "2024-06-17T00:00:00"
end_time = "2024-06-19T00:00:00"
cadence = 1 * u.hour

# Query AIA 171 data
result = Fido.search(a.Time(start_time, end_time),
                     a.Instrument.aia,
                     a.Level("1.5s"),
                    #  a.Wavelength(171*u.angstrom),
                     a.Sample(cadence)
                     )

print(result)

downloaded_files = Fido.fetch(result)

print(downloaded_files[0])
# print(downloaded_files[0].parent())

# Load data into SunPy maps
maps = sunpy.map.Map(sorted(downloaded_files))

# Set video parameters
fps = len(maps) / 10  # Duration: 10 seconds

# Create animation
fig = plt.figure(figsize=(8, 8))
ims = []

for aia_map in tqdm(maps):
    # Apply radial histogram equalization filter
    filtered_map = rhef(aia_map, progress=True, method="numpy")
    # filtered_map = sunpy.map.Map(rhef_data, aia_map.meta)
    timestamp = filtered_map.date.strftime('%Y-%m-%d %H:%M:%S')
    ax = plt.subplot(projection=filtered_map)
    im = filtered_map.plot(cmap='sdoaia171', norm=filtered_map.plot_settings['norm'], animated=True, autoalign=True)
    title = plt.text(0.5, 1.05, timestamp, ha="center", transform=ax.transAxes)
    ims.append([im, title])
    plt.savefig(str(title)+".png")

ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)

# Save the video
ani.save('aia171_video.mp4', writer='ffmpeg')

plt.close(fig)
