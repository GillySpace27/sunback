import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("qt5agg")
# Load CSV
num_runs = 5
output_file = f"./benchmark_results_{num_runs}.csv"

df = pd.read_csv(output_file)

# Convert columns
df["Scale"] = df["Resolution"]  # already numeric: 1, 2, 4
df["Average Time (s)"] = df["Average Time (s)"].astype(float)
df["Time StdDev (s)"] = df["Time StdDev (s)"].astype(float)
import numpy as np
# Plot
plt.figure(figsize=(5,4))
for filter_name in df["Filter Name"].unique():
    subset = df[df["Filter Name"] == filter_name]
    x = subset["Scale"]
    y = np.asarray(subset["Average Time (s)"])
    yerr = subset["Time StdDev (s)"]

    try:
        y /= y[0]
    except Exception as e:
        print(e)
    # yerr *= 0

    plt.plot(x, y, marker="o", label=filter_name.replace("run_", ""))
    # plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

plt.plot([1,2,4], [1, 4, 16], label="$N^2$", linestyle=(0, (6, 2, 1, 2, 1, 2)), c='grey', lw="2")
plt.plot([1,2,4], [1, 8, 16*4], label="$N^3$", linestyle=(0, (6, 2, 1, 2, 1, 2, 1, 2)), c='grey', lw="2")

plt.xlabel("Upscale Factor (1=original, 2=2k, 4=4k)")
plt.ylabel("Average Time (s)")
plt.ylabel("Relative Runtime")
plt.title("Filter Runtime vs. Image Size")
plt.xticks([1, 2, 4])
plt.legend(handlelength=4)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

for yscale in ["log", "linear"]:
    plt.yscale(yscale)
    plt.tight_layout()
    plt.savefig(f"./runtime_vs_size_{num_runs}_{yscale}.png")
plt.show()