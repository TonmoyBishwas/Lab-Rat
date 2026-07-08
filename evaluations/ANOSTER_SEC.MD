
Python

## Q4 — Python · Colour-harmony palettes (4 marks)

Pick a base hue h ∈ [0, 1]. Generate complementary, triadic, and analogous
palettes from that hue and display the swatches for all three.

Assumptions:
- The base hue $h$ is chosen as 0.1 (a light orange/yellow).
- Saturation (S) is fixed at 0.65 and Value (V) is fixed at 0.9 for all color conversions.
- The swatches will be displayed in a 1x3 subplot grid.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

# 1. Define base hue and parameters
h_base = 0.1
S = 0.65
V = 0.9

# 2. Calculate palette hues (ensuring wrapping with np.mod)
# Complementary: [h, h+0.5]
h_comp = [np.mod(h_base, 1.0), np.mod(h_base + 0.5, 1.0)]

# Triadic: [h, h+1/3, h+2/3]
h_triadic = [np.mod(h_base, 1.0), np.mod(h_base + 1/3, 1.0), np.mod(h_base + 2/3, 1.0)]

# Analogous: [h, h+1/12, h-1/12]
h_analogous = [np.mod(h_base, 1.0), np.mod(h_base + 1/12, 1.0), np.mod(h_base - 1/12, 1.0)]

# 3. Function to convert hues to RGB swatches
def build_swatches(hues, title, ax):
    # Convert hues to RGB using fixed S and V
    cols = [color.hsv2rgb([h, S, V]) for h in hues]
    
    # Draw swatches
    for i, c in enumerate(cols):
        # Draw a 1x1 rectangle patch for each color
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
    
    # Set plot limits and remove axes for clean swatch display
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title)

# 4. Display the swatches in a 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot Complementary Palette
build_swatches(h_comp, 'Complementary Palette', axes[0])

# Plot Triadic Palette
build_swatches(h_triadic, 'Triadic Palette', axes[1])

# Plot Analogous Palette
build_swatches(h_analogous, 'Analogous Palette', axes[2])

plt.suptitle(f"Color Palettes based on Base Hue h={h_base:.2f}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

Notes:
The hue arithmetic uses `np.mod(h, 1.0)` to ensure colors wrap correctly across the [0, 1] range.
The swatches are drawn using `plt.Rectangle` patches on separate axes objects to maintain control over the figure layout.
The fixed S and V values ensure that the palettes maintain a consistent saturation and brightness level.


#output: "H:\Pictures\Screenshots\Screenshot 2026-06-15 205600.png"

## Q5 — Python · RGB and HSI channel display (4 marks)

Load a bundled scikit-image sample image (`skimage.data.coffee()`). Show
its R, G, B channels as three greyscale images, and its H, S, I channels
as three more.

Assumptions:
- The image loaded is `skimage.data.coffee()`, which has dimensions (512, 512, 3).
- The output requires 6 grayscale images arranged in a grid: R, G, B (RGB channels), and H, S, I (HSV/Intensity channels).

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.util import img_as_float

# 1. Load and convert the image
img = data.coffee()
img_float = img_as_float(img)

# 2. Convert RGB to HSV
hsv = color.rgb2hsv(img_float)

# 3. Extract channels
# RGB Channels
R = img_float[..., 0]
G = img_float[..., 1]
B = img_float[..., 2]

# HSV/Intensity Channels
H = hsv[..., 0]
S = hsv[..., 1]
# Intensity (I) is the mean of R, G, B
I = img_float[..., 0:3].mean(axis=-1)

# 4. Plotting the results in a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Channel Visualization: RGB vs. HSI/Intensity", fontsize=16)

# Row 1: RGB Channels
axes[0, 0].imshow(R, cmap='gray')
axes[0, 0].set_title('Red Channel (R)')
axes[0, 0].axis('off')

axes[0, 1].imshow(G, cmap='gray')
axes[0, 1].set_title('Green Channel (G)')
axes[0, 1].axis('off')

axes[0, 2].imshow(B, cmap='gray')
axes[0, 2].set_title('Blue Channel (B)')
axes[0, 2].axis('off')

# Row 2: HSI/Intensity Channels
axes[1, 0].imshow(H, cmap='gray')
axes[1, 0].set_title('Hue Channel (H)')
axes[1, 0].axis('off')

axes[1, 1].imshow(S, cmap='gray')
axes[1, 1].set_title('Saturation Channel (S)')
axes[1, 1].axis('off')

axes[1, 2].imshow(I, cmap='gray')
axes[1, 2].set_title('Intensity Channel (I)')
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

Notes:
- All channels are displayed using `cmap='gray'` as requested for greyscale output.
- Intensity (I) was calculated as the mean of the R, G, B channels, which is the standard definition for HSI intensity.
- The use of `img_as_float` is mandatory before calling `color.rgb2hsv` to ensure correct hue values in the [0, 1] range.


#output: "H:\Pictures\Screenshots\Screenshot 2026-06-15 205929.png"

