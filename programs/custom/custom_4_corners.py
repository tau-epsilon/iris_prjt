import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Contributed by Norin
# Load the image in grayscale
image = cv2.imread('Eye_4.png', cv2.IMREAD_GRAYSCALE)

# Shave boundaries
pixels_to_shave = 10
image = image[:-pixels_to_shave, :]

# Check if the image is loaded
if image is None:
    print("Error: Could not load image.")
    exit()

# Compute the center coordinates
height, width = image.shape
center_y, center_x = height // 2, width // 2
print(center_y,',',center_x)

# Define the radius of the region to focus on
radius = 80

# Create a mask for the circular region around the center
y, x = np.indices((height, width))
mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

# Apply the mask to the image
masked_pixels = image[mask]

# Compute variance and standard deviation of the masked pixels
pixels = image.flatten()
lowest_intensities = np.partition(pixels, 20)[:60]
variance = np.var(lowest_intensities)
std_deviation = np.std(lowest_intensities)

weight = 5

# Compute minimum intensity value to compare with
min_intensity = np.min(masked_pixels)
min_intensity += variance
min_intensity += weight

# Get the coordinates of all pixels within the mask with the minimum intensity
coords = np.column_stack(np.where((image < min_intensity) & mask))

# Find the leftmost and rightmost coordinates within the masked area
if len(coords) > 0:
    leftmost = coords[np.argmin(coords[:, 1])]  # Leftmost pixel
    rightmost = coords[np.argmax(coords[:, 1])]  # Rightmost pixel
    topmost = coords[np.argmin(coords[:, 0])]  # Topmost pixel
    bottommost = coords[np.argmax(coords[:, 0])]  # Bottommost pixel

#find closest to centre
center = np.array([height // 2, width // 2])  # Center of the image (y, x)
def distance_from_center(point, center):
      return np.linalg.norm(point - center)

# Find the point closest to the center from either topmost or bottommost
top_bottom_points = np.array([topmost, bottommost])
closest_vertical = top_bottom_points[np.argmin([distance_from_center(p, center) for p in top_bottom_points])]

# Find the point closest to the center from either leftmost or rightmost
left_right_points = np.array([leftmost, rightmost])
closest_horizontal = left_right_points[np.argmin([distance_from_center(p, closest_vertical) for p in left_right_points])]

#extract pixel row
pixel_array = image[closest_horizontal[0], : ]
pixel_array = pixel_array.reshape(-1, 1)

#apply kmeans
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixel_array)
labels = kmeans.labels_

label_of_interest = labels[closest_vertical[1]]
iris_indices=np.where(labels == label_of_interest)
iris_index_first=iris_indices[0][0]
iris_index_last=iris_indices[0][-1]
ind=iris_index_last if iris_index_last-closest_vertical[1]<closest_vertical[1]-iris_index_first else iris_index_first
centre_coord=int((closest_vertical[1]+ind)/2)
print(centre_coord)
iris_rad=min(iris_index_last-closest_vertical[1],closest_vertical[1]-iris_index_first)
diff=abs((closest_vertical[0]-closest_horizontal[0]))-abs((closest_horizontal[1]-closest_vertical[1]))
diff*=1 if (closest_vertical[0]-closest_horizontal[0])>(closest_horizontal[1]-closest_vertical[1]) else -1
print('diff=',diff)
# Display results
print(f"Minimum intensity value: {min_intensity}")
print(f"Leftmost pixel with minimum intensity: {leftmost}")
print(f"Rightmost pixel with minimum intensity: {rightmost}")
print(f"Topmost pixel: {topmost}")
print(f"Bottommost pixel: {bottommost}")
print(f"Closest vertical point (topmost or bottommost): {closest_vertical}")
print(f"Closest horizontal point (leftmost or rightmost): {closest_horizontal}")

# Mark the leftmost and rightmost points on the image for visualization
image_with_points = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored markings
cv2.circle(image_with_points, (leftmost[1], leftmost[0]), 5, (0, 0, 255), -1)  # Red circle for leftmost
cv2.circle(image_with_points, (rightmost[1], rightmost[0]), 5, (255, 0, 0), -1)  # Blue circle for rightmost
cv2.circle(image_with_points, (topmost[1], topmost[0]), 5, (0, 255, 0), -1)  # Green circle at the topmost point
cv2.circle(image_with_points, (bottommost[1], bottommost[0]), 5, (0, 0, 255), -1)  # Red circle at the bottommost point
cv2.circle(image_with_points, (closest_vertical[1], closest_horizontal[0]), 5, (255, 0, 255), -1)#iris centre
cv2.circle(image_with_points, (center_x, center_y), 5, (255, 255, 255), -1)#image centre
cv2.circle(image_with_points, (closest_vertical[1]+int(diff/2),closest_horizontal[0]), iris_rad, (255, 255, 255), 1)#iris circle
# Convert BGR image to RGB for displaying in matplotlib
image_with_points_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)

# Use matplotlib to display the image in Google Colab
plt.imshow(image_with_points_rgb)
plt.title("Leftmost and Rightmost Minimum Intensity Pixels in Central Region")
plt.show()
