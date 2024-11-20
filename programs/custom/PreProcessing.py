import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
image = cv2.imread("eye.jpg", cv2.IMREAD_GRAYSCALE)
image2 = image.copy()

# Get image dimensions
height, width = image.shape

# Compute horizontal gradient
for i in range(height):
    for j in range(width - 1):
        image2[i][j] = abs(int(image[i][j + 1]) - int(image[i][j]))

# Erosion
kernel = np.ones((3, 3), np.uint8)
eroded_image = cv2.erode(image2, kernel, iterations=1)

# Dilation
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# Divide the image into 3x3 segments
segment_size = 3
threshold = 1.0000000000000001  # Set your threshold for low pixel density
output_image = dilated_image.copy()

for i in range(0, height, segment_size):
    for j in range(0, width, segment_size):
        # Define the segment
        segment = dilated_image[i:i + segment_size, j:j + segment_size]
        diff_arr=
        # Calculate the average intensity of the segment
        avg_intensity = np.mean(segment)

        # Make the segment black if the average intensity is below the threshold
        if avg_intensity < threshold:
            output_image[i:i + segment_size, j:j + segment_size] = 0

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Horizontal Gradient')
plt.imshow(image2, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Dilated Image')
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Segmented Output')
plt.imshow(output_image, cmap='gray')
plt.axis('off')

plt.show()