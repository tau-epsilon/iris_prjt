import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread("eye.jpg", cv2.IMREAD_GRAYSCALE)
pixels = image.flatten()
lowest_intensities = np.partition(pixels, 20)[:20]
variance = np.var(lowest_intensities)
std_deviation = np.std(lowest_intensities)

weight = 5

# Find the minimum intensity (lowest intensity pixel value)
min_intensity = np.min(image)
min_intensity += variance
min_intensity += weight

# Get the coordinates of all pixels with the minimum intensity
coords = np.column_stack(np.where(image < min_intensity))

# Find the leftmost and rightmost coordinates
leftmost = coords[np.argmin(coords[:, 1])]  # Leftmost pixel
rightmost = coords[np.argmax(coords[:, 1])]  # Rightmost pixel

# Find the topmost and bottommost coordinates
topmost = coords[np.argmin(coords[:, 0])]  # Topmost pixel
bottommost = coords[np.argmax(coords[:, 0])]  # Bottommost pixel

# Find the two closest points: one from the topmost/bottommost and one from the leftmost/rightmost
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# We need to find the closest pair between topmost/bottommost and leftmost/rightmost points
distances = [
    (topmost, leftmost), (topmost, rightmost), 
    (bottommost, leftmost), (bottommost, rightmost)
]

# Find the closest pair of points
closest_pair = min(distances, key=lambda pair: euclidean_distance(pair[0], pair[1]))

# Get the coordinates of the closest pair
point1, point2 = closest_pair

# Calculate the intersection point (the midpoint between the two closest points)
intersection_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

# Mark the points on the image for visualization
image_with_points = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored markings
cv2.circle(image_with_points, (leftmost[1], leftmost[0]), 5, (0, 0, 255), -1)  # Red circle for leftmost
cv2.circle(image_with_points, (rightmost[1], rightmost[0]), 5, (255, 0, 0), -1)  # Blue circle for rightmost
cv2.circle(image_with_points, (topmost[1], topmost[0]), 5, (0, 255, 0), -1)  # Green circle for topmost
cv2.circle(image_with_points, (bottommost[1], bottommost[0]), 5, (0, 255, 255), -1)  # Yellow circle for bottommost
cv2.circle(image_with_points, (intersection_point[1], intersection_point[0]), 5, (255, 255, 255), -1)  # White circle for intersection

# Convert BGR image to RGB for displaying in matplotlib
image_with_points_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)

# Show the image with points and intersection marked
plt.imshow(image_with_points_rgb)
plt.axis('off')  # Hide axes for better visualization
plt.tight_layout()
plt.show()



#Red Circle: Leftmost point
#Blue Circle: Rightmost point
#Green Circle: Topmost point
#Yellow Circle: Bottommost point
#White Circle: Intersection point (midpoint between the closest points)