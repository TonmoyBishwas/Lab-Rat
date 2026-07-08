# Lab Rat AI — Python (Lab 04) model responses

Paste the full text of each model response under its heading.
Question text lives in `python_lab04_questions.md`.

---

## Q1 — RGB and HSV channel display

<!-- paste response here -->

---Assumptions:
- The user is referring to the standard `skimage.data.coffee()` dataset.
- The required output is a 2x3 grid of greyscale images.

```python
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import skimage.color
import skimage.util

# Load the image and convert to float
img_uint8 = skimage.data.coffee()
img = skimage.util.img_as_float(img_uint8)

# 1. RGB Channels
R, G, B = img[..., 0], img[..., 1], img[..., 2]

# 2. HSV Channels
hsv = skimage.color.rgb2hsv(img)
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

# Organize channels for plotting
channels = {
    'R': R, 'G': G, 'B': B,
    'H': H, 'S': S, 'V': V
}
channel_names = list(channels.keys())

# Create the 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten() # Flatten the 2x3 array of axes for easy iteration

# Plot each channel
for i, name in enumerate(channel_names):
    ax = axes[i]
    channel_data = channels[name]
    
    # Display the channel as a greyscale image
    ax.imshow(channel_data, cmap='gray')
    ax.set_title(f'{name} Channel')
    ax.axis('off') # Hide axes ticks and labels

plt.suptitle("Coffee Image Channels (RGB and HSV)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()
```

Notes:
- We used `img_as_float()` before conversion to ensure correct color space handling by `rgb2hsv`.
- All channels (R, G, B, H, S, V) are displayed using `cmap='gray'` as requested, even though H, S, V are floating-point values in [0, 1].
- The axes were flattened to simplify the loop structure for plotting the 6 subplots.

## Q2 — Three colour-harmony palettes from one hue

<!-- paste response here -->
Assumptions:
- The user requires a 3x1 subplot grid.
- The base hue is $h=0.55$.
- Saturation is $S=0.65$ and Value is $V=0.9$.

```python
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
from matplotlib.patches import Rectangle

# --- Parameters ---
h_base = 0.55
S = 0.65
V = 0.9

# --- Hue Calculations ---
# 1. Complementary (2 swatches)
h_comp = np.array([h_base, h_base + 0.5])
h_comp = np.mod(h_comp, 1.0)
cols_comp = [skimage.color.hsv2rgb([h, S, V]) for h in h_comp]

# 2. Triadic (3 swatches)
h_tri = np.array([h_base, h_base + 1/3, h_base + 2/3])
h_tri = np.mod(h_tri, 1.0)
cols_tri = [skimage.color.hsv2rgb([h, S, V]) for h in h_tri]

# 3. Analogous (3 swatches)
h_ana = np.array([h_base, h_base + 1/12, h_base - 1/12])
h_ana = np.mod(h_ana, 1.0)
cols_ana = [skimage.color.hsv2rgb([h, S, V]) for h in h_ana]

# --- Plotting ---
fig, axes = plt.subplots(3, 1, figsize=(6, 4))
plt.suptitle(f"Color Palettes based on Base Hue h={h_base}", fontsize=14)

# 1. Complementary Palette
ax_comp = axes[0]
for i, c in enumerate(cols_comp):
    ax_comp.add_patch(Rectangle((i, 0), 1, 1, color=c))
ax_comp.set_xlim(0, len(cols_comp))
ax_comp.set_ylim(0, 1)
ax_comp.set_title("1. Complementary (2 swatches)")
ax_comp.axis('off')

# 2. Triadic Palette
ax_tri = axes[1]
for i, c in enumerate(cols_tri):
    ax_tri.add_patch(Rectangle((i, 0), 1, 1, color=c))
ax_tri.set_xlim(0, len(cols_tri))
ax_tri.set_ylim(0, 1)
ax_tri.set_title("2. Triadic (3 swatches)")
ax_tri.axis('off')

# 3. Analogous Palette
ax_ana = axes[2]
for i, c in enumerate(cols_ana):
    ax_ana.add_patch(Rectangle((i, 0), 1, 1, color=c))
ax_ana.set_xlim(0, len(cols_ana))
ax_ana.set_ylim(0, 1)
ax_ana.set_title("3. Analogous (3 swatches)")
ax_ana.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

Notes:
- The swatches are created using `matplotlib.patches.Rectangle` and added directly to the axes object.
- `np.mod(h, 1.0)` is crucial to ensure all calculated hues wrap correctly within the $[0, 1]$ range required by `skimage.color.hsv2rgb`.
- The `plt.tight_layout` call is adjusted to prevent the `suptitle` from overlapping the top subplot.

---

## Q3 — Manual violin plot of iris petal length

<!-- paste response here -->
Assumptions:
- The `iris` dataset is available via `seaborn.load_dataset('iris')`.
- The relevant columns are 'petal_length' (the value) and 'species' (the group).
- There are three unique species in the dataset.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the iris dataset
iris = sns.load_dataset('iris')

# Identify unique species and values
species_list = iris['species'].unique()
num_species = len(species_list)

# Setup figure and axes for the manual violin plot
fig, ax = plt.subplots(figsize=(10, 6))

# Loop through each species to draw the violin
for i, species in enumerate(species_list):
    # Get petal lengths for the current species
    vals = iris[iris['species'] == species]['petal_length'].values

    # Calculate Gaussian Kernel Density Estimate (KDE)
    kde = stats.gaussian_kde(vals)

    # Define the range for the x-axis (the value axis)
    # Extend the range slightly beyond min/max for better visualization
    x_min = vals.min() - 0.1
    x_max = vals.max() + 0.1
    xs = np.linspace(x_min, x_max, 300)

    # Calculate the density function values
    dens = kde(xs)

    # Normalize density and scale it for visualization (e.g., max height 0.4)
    dens = dens / dens.max() * 0.4

    # Draw the violin shape using fill_betweenx
    # Critical argument order: xs (value axis), i - dens (lower bound), i + dens (upper bound)
    ax.fill_betweenx(xs, i - dens, i + dens, alpha=0.5, label=species)

# Set x-ticks and labels
ax.set_xticks(range(num_species))
ax.set_xticklabels(species_list)

# Add titles and labels
ax.set_title('Manual Violin Plot of Petal Length by Species')
ax.set_xlabel('Species')
ax.set_ylabel('Petal Length')

# Clean up the plot appearance
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

Notes:
- The manual KDE approach requires careful handling of the `fill_betweenx` arguments, ensuring `xs` is the first argument (the value axis).
- If the data distribution is highly skewed, adjusting the `alpha` value or the scaling factor (`* 0.4`) may be necessary to improve visual clarity.
- For a more robust visualization, consider adding vertical lines or markers at the median/mean within each violin shape.
---

## Q4 — Divergent heatmap of iris correlation matrix

<!-- paste response here -->
Assumptions:
- The `iris` dataset contains four numeric columns suitable for correlation calculation (sepal_length, sepal_width, petal_length, petal_width).
- The required libraries (numpy, pandas, matplotlib, seaborn, scipy, skimage) are available in the environment.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the dataset
iris = sns.load_dataset("iris")

# Calculate the correlation matrix for numeric columns
corr_matrix = iris.select_dtypes(include=np.number).corr()

# Define the divergent colormap (Blue -> White -> Red)
# We name the map 'bwr_div' but pass it as a variable.
stops = [(0.0, 'blue'), (0.5, 'white'), (1.0, 'red')]
div_cmap = LinearSegmentedColormap.from_list('bwr_div', stops)

# Plot the heatmap
plt.figure(figsize=(8, 7))
sns.heatmap(
    corr_matrix,
    cmap=div_cmap,
    center=0,  # Center the colormap at 0 for divergence
    annot=True, # Annotate each cell with the correlation value
    fmt=".2f",  # Format annotations to two decimal places
    linewidths=.5, # Add lines between cells for clarity
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title("Correlation Matrix of Iris Features (Divergent Colormap)")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
```

Notes:
- Using `center=0` is crucial for the divergent colormap to correctly map positive correlations to red and negative correlations to blue.
- The custom colormap must be built using `LinearSegmentedColormap.from_list` and passed as the `cmap` argument, not referenced by name as a string.
- `annot=True` and `fmt=".2f"` ensure the correlation values are clearly displayed in each cell.
---

Assumptions:
- The `iris` dataset is available via `seaborn.load_dataset`.
- All four features in `iris` ('sepal_length', 'sepal_width', 'petal_length', 'petal_width') are numeric and suitable for correlation calculation.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the iris dataset
iris = sns.load_dataset('iris')

# Calculate the correlation matrix for numeric columns
corr_matrix = iris.select_dtypes(include=np.number).corr()

# Define the divergent colormap stops (Blue -> White -> Red)
stops = [(0.0, 'blue'), (0.5, 'white'), (1.0, 'red')]

# Build the custom divergent colormap variable
div_cmap = LinearSegmentedColormap.from_list('bwr_div', stops)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    cmap=div_cmap,
    center=0,  # Center the colormap at 0
    annot=True, # Annotate each cell with the correlation value
    fmt=".2f",  # Format annotations to two decimal places
    linewidths=.5, # Add lines between cells for clarity
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title('Correlation Matrix of Iris Features (Divergent Colormap)')
plt.xlabel('Feature')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
```

Notes:
- Using `sns.heatmap` simplifies the visualization of the correlation matrix.
- Setting `center=0` ensures the colormap transitions correctly from blue (negative) through white (zero) to red (positive).
- The `LinearSegmentedColormap.from_list` method is used to create the required custom divergent colormap variable.

## Q5 — All four outlier-handling techniques side-by-side
Assumptions:
- The matrix $M$ is a $50 \times 50$ NumPy array containing noisy data with planted outliers.
- The visualization requires a $2 \times 2$ grid of subplots.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# 1. Generate the noisy matrix M
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20   # planted outliers

# 2. Calculate the transformed matrices
# Technique 1: Clip
q_low = np.quantile(M, 0.02)
q_high = np.quantile(M, 0.98)
M_clip = np.clip(M, q_low, q_high)

# Technique 2: Quantile-rank normalisation
ranks = M.flatten().argsort().argsort() / (M.size - 1)
M_quantile_rank = ranks.reshape(M.shape)

# Technique 3: Signed log1p
M_signed_log1p = np.sign(M) * np.log1p(np.abs(M))

# Technique 4: Alpha overlay (M itself)

# 3. Setup the 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
plt.suptitle("Outlier Handling Techniques on Noisy Matrix M", fontsize=16)

# Define a common colormap for consistency
cmap = 'viridis'

# Plot 1: Clip
im1 = axes[0, 0].imshow(M_clip, cmap=cmap)
axes[0, 0].set_title("1. Clip (2nd-98th Quantile)")
axes[0, 0].set_xlabel("Column Index")
axes[0, 0].set_ylabel("Row Index")
fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

# Plot 2: Quantile-rank normalisation
im2 = axes[0, 1].imshow(M_quantile_rank, cmap=cmap)
axes[0, 1].set_title("2. Quantile-Rank Normalisation")
axes[0, 1].set_xlabel("Column Index")
axes[0, 1].set_ylabel("Row Index")
fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

# Plot 3: Signed log1p
im3 = axes[1, 0].imshow(M_signed_log1p, cmap=cmap)
axes[1, 0].set_title("3. Signed Log1p")
axes[1, 0].set_xlabel("Column Index")
axes[1, 0].set_ylabel("Row Index")
fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')

# Plot 4: Alpha overlay (using original M)
# Note: We use ax.imshow for the alpha technique
im4 = axes[1, 1].imshow(M, cmap=cmap, alpha=0.85)
axes[1, 1].set_title("4. Alpha Overlay (Original M)")
axes[1, 1].set_xlabel("Column Index")
axes[1, 1].set_ylabel("Row Index")
fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()
```

Notes:
- All techniques are visualized using `imshow` since the data is a matrix.
- The alpha overlay technique is specifically demonstrated by setting the `alpha` parameter within `imshow`.
- If the data were continuous and 1D, a heatmap would be replaced by a density plot or a line plot.
<!-- paste response here -->

---

## Q6 — Multi-checkpoint quantile-anchored colormap

<!-- paste response here -->
Assumptions:
- The matrix $M$ is a $10 \times 10$ matrix of synthetic random data, representing the matrix from Q5.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. Synthesize the matrix M (10x10)
np.random.seed(42)
M = np.random.rand(10, 10) * 10 - 5 # Range roughly [-5, 5]

# 2. Calculate vmin and vmax
vmin = M.min()
vmax = M.max()

# 3. Calculate quantiles (5th, 50th, 95th)
qs = [0.05, 0.50, 0.95]
qvals = np.quantile(M, qs).flatten()

# 4. Define colors
colors = ['navy', 'white', 'darkred']

# 5. Calculate normalized positions
# Normalization formula: (q - vmin) / (vmax - vmin)
pos_list = [(q - vmin) / (vmax - vmin) for q in qvals]

# 6. Create the list of stops: [(position, color), ...]
# We must ensure the order matches the quantiles: 5th (navy), 50th (white), 95th (darkred)
stops_raw = list(zip(pos_list, colors))

# 7. Deduplicate stops based on position (within 1e-6)
# Sort by position to make deduplication easier
stops_raw.sort(key=lambda x: x[0])

stops_deduplicated = []
if stops_raw:
    stops_deduplicated.append(stops_raw[0])
    for i in range(1, len(stops_raw)):
        current_pos, current_color = stops_raw[i]
        last_pos, _ = stops_deduplicated[-1]
        
        # Check if the position is significantly different from the last added stop
        if abs(current_pos - last_pos) > 1e-6:
            stops_deduplicated.append(stops_raw[i])

# 8. Build the LinearSegmentedColormap
cmap_name = 'custom_quantile_map'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, stops_deduplicated)

# 9. Generate the heatmap
plt.figure(figsize=(8, 7))
sns.heatmap(M, cmap=custom_cmap, vmin=vmin, vmax=vmax, annot=False)
plt.title("Heatmap with Quantile-Anchored Piecewise Colormap")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.tight_layout()
plt.show()
```

Notes:
- The `LinearSegmentedColormap.from_list` function requires a list of (position, color) tuples, which we constructed using the normalized quantiles.
- Passing `vmin` and `vmax` to `sns.heatmap` ensures the color mapping is correctly scaled to the full range of the data, regardless of the specific quantile anchors.
- If your matrix $M$ has non-uniform distribution, the visual impact of the colormap will change, but the mathematical anchoring remains correct.
---

## Q7 — Ridgeline of 8 groups inside a 2×4 subplot grid

<!-- paste response here -->
Assumptions:
- The data will be generated as a single column of values, grouped by 8 categorical labels (A through H).
- Each group contains exactly 200 samples.
- The means of the 8 groups will be distinct and increasing for clear visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# --- Data Generation ---
groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
data_dict = {}
N_samples = 200
means = np.arange(10, 10 + len(groups)) * 2

for i, group_name in enumerate(groups):
    # Generate 200 samples from a normal distribution with a specific mean
    data = np.random.normal(loc=means[i], scale=3, size=N_samples)
    data_dict[group_name] = data

# --- Plotting Setup ---
fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False, sharey=False)
axes = axes.flatten() # Flatten the 2x4 array into a 1D array for easy iteration

# --- Manual Ridgeline (Stacked KDE) Implementation ---
for i, group_name in enumerate(groups):
    ax = axes[i]
    vals = data_dict[group_name]

    # 1. Calculate KDE
    kde = scipy.stats.gaussian_kde(vals)
    
    # 2. Define the value axis range
    xs = np.linspace(vals.min(), vals.max(), 200)
    
    # 3. Calculate density
    dens = kde(xs)
    
    # 4. Normalize density and scale for visualization (alpha=0.4)
    # We normalize by the max density and scale it down to ensure overlap works well
    dens = dens / dens.max() * 0.4
    
    # 5. Fill the area (Manual Ridgeline)
    # CRITICAL ARG ORDER: xs (value axis), i - dens (lower bound), i + dens (upper bound)
    ax.fill_between(xs, -dens, dens, alpha=0.5, color='skyblue')
    
    # Set titles and labels for the subplot
    ax.set_title(f'Ridgeline for Group {group_name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent overlap
plt.suptitle("Ridgeline Visualization of 8 Groups (A-H)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

Notes:
- Since the question requires a ridgeline inside a subplot grid, the manual stacked-KDE fallback was used instead of `joypy.joyplot`.
- The `fill_between` function is used to draw the density curve, centered around zero, simulating the ridge shape.
- If the data were highly skewed, adjusting the `scale` parameter in `np.random.normal` or changing the `alpha` value in `fill_between` would be necessary for better visual separation.
---

## Q8 — Geometric mean and contra-harmonic mean of a vector

<!-- paste response here -->

Assumptions:
- The input array `x` is `[1, 2, 4, 8]`.

```python
import numpy as np

# Input data
x = np.array([1, 2, 4, 8])

# 1. Geometric Mean (GM)
# Formula: np.exp(np.log(x).mean())
gm = np.exp(np.log(x).mean())

# 2. Contra-Harmonic Mean (CHM, p=1)
# Formula: (x**2).sum() / x.sum()
chm = (x**2).sum() / x.sum()

print(f"1. Geometric Mean: {gm:.4f}")
print(f"2. Contra-Harmonic Mean (p=1): {chm:.4f}")
```

Notes:
- The geometric mean is suitable for positive data, which `x` is.
- The contra-harmonic mean is calculated directly using the provided formula for p=1.
- Both calculations are straightforward and require no complex plotting or data manipulation.