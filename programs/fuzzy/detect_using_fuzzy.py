import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io, img_as_float
from scipy.ndimage import convolve

# Load the image and convert to grayscale
path = "/home/idiot/Downloads/1-543.JPG"
Irgb = io.imread(path)  # Replace with actual path to the image
Igray = color.rgb2gray(Irgb)
I = img_as_float(Igray)

# Define Sobel filters for x and y gradients
Gx = np.array([[-1, 1]])
Gy = Gx.T
Ix = convolve(I, Gx, mode='reflect')
Iy = convolve(I, Gy, mode='reflect')

# Define Gaussian membership functions for Ix and Iy (zero region)
sx, sy = 0.1, 0.1
Ix_zero = np.exp(-((Ix / sx) ** 2) / 2)
Iy_zero = np.exp(-((Iy / sy) ** 2) / 2)

# Create the output based on fuzzy rules without looping
# Rule 1: If Ix is zero and Iy is zero then Iout is white (1)
# Rule 2: If Ix is not zero or Iy is not zero then Iout is black (0)
Iout_white = Ix_zero * Iy_zero   # Both Ix and Iy near zero -> white
Iout_black = 1 - Iout_white       # Either Ix or Iy non-zero -> black
Ieval = Iout_white

# Display results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(Igray, cmap='gray')
axs[0].set_title("Original Grayscale Image")
axs[1].imshow(Ieval, cmap='gray')
axs[1].set_title("Edge Detection Using Optimized Fuzzy Logic")
axs[2].imshow(I, cmap='gray')
axs[2].set_title("Gradient Output")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
