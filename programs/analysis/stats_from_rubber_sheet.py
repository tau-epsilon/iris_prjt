import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis, variation
from statistics import mode
import plotly.graph_objs as go

# Function to generate the rubber sheet model (polar unwrapping)
def generate_rubber_sheet_model(img):
    q = np.arange(0.00, np.pi / 3, 0.001)
    inn = np.arange(0, int(img.shape[0] / 2), 1)
    cartisian_image = np.empty(shape=[inn.size, int(img.shape[1]), 3])

    m = interp1d([np.pi * 2, 0], [0, img.shape[1]])

    for r in inn:
        for t in q:
            polarX = int((r * np.cos(t)) + img.shape[1] / 2)
            polarY = int((r * np.sin(t)) + img.shape[0] / 2)
            if 0 <= polarX < img.shape[1] and 0 <= polarY < img.shape[0]:  # Check bounds
                cartisian_image[r][int(m(t) - 1)] = img[polarY][polarX]
    return cartisian_image.astype("uint8")

# Function to remove reflection using inpainting
def remove_reflection(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=2)
    dst = cv2.inpaint(img, dilation, 5, cv2.INPAINT_TELEA)
    return dst

# Function to process the image and detect circles
def processing(image_path, r):
    success = False
    original_image = cv2.imread(image_path)
    image = cv2.resize(original_image, (640, 480), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 11)
    ret, _ = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=ret,
        param2=30,
        minRadius=20,
        maxRadius=100,
    )

    if circles is not None:
        circles = circles[0, :, :]
        circles = np.int16(np.array(circles))
        for i in circles[:]:
            image_roi = image[
                i[1] - i[2] - r : i[1] + i[2] + r, i[0] - i[2] - r : i[0] + i[2] + r
            ]
            radius = i[2]
        success = True
        return image_roi, radius, success
    else:
        print(f"{image_path} -> No circles (iris) found.")
        return None, None, success

# Updated function to handle precision loss in statistics
def compute_rubber_sheet_statistics(rubber_sheet):
    means, medians, modes, skewnesses, kurtoses = [], [], [], [], []
    for row in rubber_sheet[..., 0]:  # Assuming grayscale or red channel
        row_data = row[row > 0]  # Exclude any zero-padded areas
        if len(row_data) > 0:
            means.append(np.mean(row_data))
            medians.append(np.median(row_data))
            try:
                modes.append(mode(row_data))
            except:
                modes.append(np.nan)  # Handle multiple modes by storing NaN

            # Check for minimal variance
            if np.var(row_data) > 1e-5:  # Small threshold for variance
                skewnesses.append(skew(row_data))
                kurtoses.append(kurtosis(row_data))
            else:
                skewnesses.append(np.nan)
                kurtoses.append(np.nan)
        else:
            means.append(np.nan)
            medians.append(np.nan)
            modes.append(np.nan)
            skewnesses.append(np.nan)
            kurtoses.append(np.nan)
    return means, medians, modes, skewnesses, kurtoses


# Main processing function
def process_img(image_path, keep_reflection=False):
    image_roi, rr, success = processing(image_path, 50)

    if success:
        if not keep_reflection:
            image_roi = remove_reflection(image_roi)

        # Generate the rubber sheet model
        rubber_sheet = generate_rubber_sheet_model(image_roi)

        # Compute statistics for each row
        means, medians, modes, skewnesses, kurtoses = compute_rubber_sheet_statistics(rubber_sheet)

        # Display the computed statistics
        print("Row-wise Statistics for Rubber Sheet Model:")
        print("Means:", means)
        print("Medians:", medians)
        print("Modes:", modes)
        print("Skewness:", skewnesses)
        print("Kurtosis:", kurtoses)
    else:
        print("Iris or Pupil not detected.")

# Test the function on your image
image_path = 'eye.jpg'  # Replace with your image file
process_img(image_path)
