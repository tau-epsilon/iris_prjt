import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

# Directories for input (eyelash-free images) and output (images with detected circles)
eyelash_free_directory = (
    "/home/idiot/output_eyelashed_cleared"  # Directory with eyelash-free images
)
circle_detected_directory = (
    "/home/idiot/output_circles_detected"  # Directory to save images with circles drawn
)
os.makedirs(circle_detected_directory, exist_ok=True)

# Define Hough Circle parameters
min_radius = 20  # Minimum radius of circles to detect
max_radius = 50  # Maximum radius of circles to detect
num_radii = 30  # Number of radii to test in the Hough transform

# Process each eyelash-cleared image
for file in os.listdir(eyelash_free_directory):
    if file.endswith(".png",".jpg", ".tiff",".bmp"):
        # Load the eyelash-free image
        img_path = os.path.join(eyelash_free_directory, file)
        image = io.imread(img_path, as_gray=True)

        # Perform edge detection (Canny) for Hough Transform
        edges = canny(image, sigma=2)

        # Perform Hough Circle Transform
        hough_radii = np.linspace(min_radius, max_radius, num_radii)
        hough_res = hough_circle(edges, hough_radii)

        # Select the most prominent circles
        accums, cx, cy, radii = hough_circle_peaks(
            hough_res, hough_radii, total_num_peaks=1
        )

        # Convert the grayscale image to RGB for color overlay of circles
        image_with_circles = (
            color.gray2rgb(image) * 255
        )  # Convert to RGB and scale to 0-255

        # Draw the detected circles on the image
        for center_y, center_x, radius in zip(cy, cx, radii):
            # Ensure the coordinates and radius are integers
            center_y, center_x, radius = int(center_y), int(center_x), int(radius)
            circy, circx = circle_perimeter(
                center_y, center_x, radius, shape=image.shape
            )
            # Draw the perimeter in red (RGB: (255, 0, 0))
            image_with_circles[circy, circx] = [255, 0, 0]  # Red color for the circle

        # Save the image with detected circles
        save_path = os.path.join(circle_detected_directory, f"circle_detected_{file}")

        # Convert to the correct type for saving
        plt.imsave(
            save_path, image_with_circles.astype(np.uint8)
        )  # Ensure data type is uint8

        # Optional: Print progress
        print(f"Detected and saved circles in image: {save_path}")

print("Circle detection completed for all images.")
