## Q6 — HARD · Multi-checkpoint quantile-anchored colormap

Generate this noisy matrix (paste the block verbatim along with the question
text — the launcher does NOT remember Q5):
```python
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20
```
Build a custom **piecewise colormap** whose colour stops are anchored at the
5th, 50th, and 95th quantiles of `M` (colours: navy → white → darkred).
Normalise the stop positions to `[0, 1]` using `(q - vmin) / (vmax - vmin)`.
Deduplicate stops that are within `1e-6` of each other. Pass the cmap to
`sns.heatmap` with the same `vmin`/`vmax`.

ANSWER:
Assumptions:
- The noisy matrix $M$ is generated using `np.random.seed(0)` and the provided code block.
- The required quantiles are 5th (0.05), 50th (0.50), and 95th (0.95).
- The colors are 'navy', 'white', and 'darkred'.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 1. Generate the noisy matrix M
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20

# 2. Calculate vmin and vmax
vmin = M.min()
vmax = M.max()

# 3. Calculate required quantiles
qs = [0.05, 0.50, 0.95]
q_values = np.quantile(M, qs)

# 4. Define colors
colors = ['navy', 'white', 'darkred']

# 5. Calculate normalized positions
# Normalization formula: (q - vmin) / (vmax - vmin)
pos = [(q - vmin) / (vmax - vmin) for q in q_values]

# 6. Create the list of stops
stops = list(zip(pos, colors))

# 7. Deduplicate stops (required by prompt, though unlikely to change anything here)
# We sort by position and then filter out duplicates within 1e-6
stops.sort(key=lambda x: x[0])
deduplicated_stops = []
if stops:
    deduplicated_stops.append(stops[0])
    for i in range(1, len(stops)):
        prev_pos, prev_color = stops[i-1]
        curr_pos, curr_color = stops[i]
        if abs(curr_pos - prev_pos) > 1e-6:
            deduplicated_stops.append(stops[i])
stops = deduplicated_stops

# 8. Build the custom piecewise colormap
custom_cmap = LinearSegmentedColormap.from_list('quantile_anchored_cmap', stops)

# 9. Plot the heatmap
plt.figure(figsize=(8, 7))
sns.heatmap(M, cmap=custom_cmap, vmin=vmin, vmax=vmax, annot=False)
plt.title("Heatmap with Quantile-Anchored Piecewise Colormap")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.tight_layout()
plt.show()
```

Notes:
The `LinearSegmentedColormap.from_list` function handles the piecewise definition based on the provided stops.
Ensure `vmin` and `vmax` are passed to `sns.heatmap` to correctly map the data values to the colormap range.
If the matrix $M$ were highly uniform, the normalized positions might cluster, making the deduplication step more relevant.



Generate synthetic data for 8 groups (A–H), each with 200 samples drawn from a
normal distribution with a different mean. Plot a **ridgeline**: 8 stacked KDE
curves on a SINGLE set of axes, each curve vertically offset by its group
index, with group names on the y-axis tick labels. Do this without
`joypy.joyplot`. Use `scipy.stats.gaussian_kde` and the stacked
`fill_between(xs, i - dens, i + dens, ...)` pattern from the prompt.


#ANSWER:
Assumptions:
- 8 groups are labeled A through H.
- Each group has 200 samples.
- The means for the 8 groups are distinct (e.g., 10, 12, 14, 16, 18, 20, 22, 24).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# 1. Generate synthetic data
N_samples = 200
group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
means = np.arange(10, 10 + 8) * 2

data = {}
for i, name in enumerate(group_names):
    # Generate 200 samples from a normal distribution with a distinct mean
    data[name] = np.random.normal(loc=means[i], scale=5, size=N_samples)

# 2. Setup plot and iterate through groups
fig, ax = plt.subplots(figsize=(10, 6))

# Determine the overall range for the x-axis (value axis)
all_data = np.concatenate(list(data.values()))
min_val = all_data.min() - 5
max_val = all_data.max() + 5
xs = np.linspace(min_val, max_val, 200)

# Iterate through groups and plot stacked KDEs
for i, group_name in enumerate(group_names):
    vals = data[group_name]
    
    # Calculate KDE
    kde = scipy.stats.gaussian_kde(vals)
    
    # Calculate density on the defined x-axis range
    dens = kde(xs)
    
    # Draw the stacked KDE curve
    # Note: The vertical offset is achieved by shifting the density bounds
    ax.fill_between(xs, i - dens, i + dens, alpha=0.4, label=group_name)

# 3. Set y-axis labels and ticks
ax.set_yticks(range(len(group_names)))
ax.set_yticklabels(group_names)
ax.set_ylabel("Group Index")
ax.set_xlabel("Value")
ax.set_title("Ridgeline Plot of Group Distributions (Manual Stacked KDE)")

# Optional: Add a legend if desired, though the y-axis labels are primary
# ax.legend(loc='upper right') 

plt.tight_layout()
plt.show()
```

Notes:
- The vertical offset is achieved by using the index `i` (the group index) to shift the density bounds (`i - dens` and `i + dens`) in `fill_between`.
- The x-axis range (`xs`) must be calculated based on the minimum and maximum values across all groups to ensure all curves are plotted correctly.
- Use `ax.fill_between` (the axes object) rather than `plt.fill_between_x` to ensure the vertical stacking works correctly.


## Q5 — HARD · All four outlier-handling techniques side-by-side
Generate a noisy 50×50 matrix:
```
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20   # planted outliers
```
Plot a 2×2 subplot grid showing the **four outlier-handling techniques** from
Lab 04 Part C, each applied to `M`:

1. Clip to the 2nd–98th quantile
2. Quantile-rank normalisation
3. Signed log1p (`np.sign(M) * np.log1p(np.abs(M))`)
4. Alpha overlay (`ax.imshow(M, alpha=0.85)`)

Title each subplot with its technique name.

#ANSWER;
Assumptions:
- The matrix $M$ is a 50x50 NumPy array generated using the provided seeding and outlier injection logic.
- The four techniques (Clip, Quantile, Signed Log1p, Alpha Overlay) will be displayed in a 2x2 grid.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate the noisy matrix M
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20   # planted outliers

# --- Outlier Handling Techniques ---

# 1. Clip: Clip to the 2nd–98th quantile
M_clip = np.clip(M, np.quantile(M, 0.02), np.quantile(M, 0.98))

# 2. Quantile-rank normalisation
ranks = M.flatten().argsort().argsort() / (M.size - 1)
M_quantile = ranks.reshape(M.shape)

# 3. Signed log1p
M_signed_log1p = np.sign(M) * np.log1p(np.abs(M))

# 4. Alpha overlay (M itself, displayed with alpha)
# This technique is applied directly in the plotting step

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
fig.suptitle("Outlier Handling Techniques on Noisy Matrix M", fontsize=16)

# 1. Clip
ax00 = axes[0, 0]
im00 = ax00.imshow(M_clip, cmap='viridis')
ax00.set_title("1. Clip (2nd-98th Quantile)")
ax00.set_xlabel("Column Index")
ax00.set_ylabel("Row Index")
fig.colorbar(im00, ax=ax00, orientation='vertical', fraction=0.046, pad=0.04)

# 2. Quantile-rank normalisation
ax01 = axes[0, 1]
im01 = ax01.imshow(M_quantile, cmap='viridis')
ax01.set_title("2. Quantile Ranking")
ax01.set_xlabel("Column Index")
ax01.set_ylabel("Row Index")
fig.colorbar(im01, ax=ax01, orientation='vertical', fraction=0.046, pad=0.04)

# 3. Signed log1p
ax10 = axes[1, 0]
im10 = ax10.imshow(M_signed_log1p, cmap='viridis')
ax10.set_title("3. Signed Log1p")
ax10.set_xlabel("Column Index")
ax10.set_ylabel("Row Index")
fig.colorbar(im10, ax=ax10, orientation='vertical', fraction=0.046, pad=0.04)

# 4. Alpha overlay (using M directly)
ax11 = axes[1, 1]
# Using Pattern B (overlay on a visible grid) for better visualization of the effect
ax11.set_xticks(np.arange(M.shape[1]) + 0.5, minor=True)
ax11.set_yticks(np.arange(M.shape[0]) + 0.5, minor=True)
ax11.grid(which='minor', color='white', linewidth=0.5)
ax11.imshow(M, cmap='viridis', alpha=0.85)
ax11.set_title("4. Alpha Overlay (M with alpha=0.85)")
ax11.set_xlabel("Column Index")
ax11.set_ylabel("Row Index")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

Notes:
- All techniques are visualized using `imshow` since the data is a matrix, showing spatial distribution.
- For the Alpha Overlay, I used Pattern B (overlay on a grid) as it clearly demonstrates the effect of the transparency on the underlying structure.
- If the matrix were 1D, a line plot or histogram would be more appropriate than `imshow`.




## Q4 — MED · Divergent heatmap of the iris correlation matrix

Build a correlation matrix of the four numeric columns of `iris`. Plot it as a
heatmap with a **divergent colormap** centred at zero (blue → white → red).
Annotate each cell with the correlation value. Build the colormap as a variable
with `LinearSegmentedColormap.from_list`; do not pass a bare string.

#ANSWER:
Assumptions:
- The `iris` dataset contains four numeric columns representing measurements (sepal length, sepal width, petal length, petal width).
- The correlation matrix calculation will be performed on these four numeric columns.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the iris dataset
iris = sns.load_dataset('iris')

# Calculate the correlation matrix for numeric columns
corr_matrix = iris.select_dtypes(include=np.number).corr()

# Define the divergent colormap stops: Blue (low), White (center), Red (high)
stops = [(0.0, 'blue'), (0.5, 'white'), (1.0, 'red')]
# Build the custom divergent colormap variable
div_cmap = LinearSegmentedColormap.from_list('bwr_div', stops)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap=div_cmap,
    center=0,  # Center the colormap at zero
    fmt=".2f", # Format annotations to two decimal places
    linewidths=.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title('Correlation Matrix of Iris Features')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
```

Notes:
- Using `center=0` ensures the colormap transitions smoothly from blue (negative correlation) through white (zero correlation) to red (positive correlation).
- Building the colormap via `LinearSegmentedColormap.from_list` and passing the variable `div_cmap` is necessary to meet the specific requirement.
- `annot=True` displays the correlation values directly on the heatmap cells.