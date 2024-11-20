import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objs as go

# Function to save and display an image using matplotlib
def display_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to generate the rubber sheet model (polar unwrapping)
def generate_rubber_sheet_model(img):
    q = np.arange(0.00, np.pi/3, 0.001)
    inn = np.arange(0, int(img.shape[0] / 2), 1)
    cartisian_image = np.empty(shape=[inn.size, int(img.shape[1]), 3])

    m = interp1d([np.pi * 2, 0], [0, img.shape[1]])

    for r in inn:
        for t in q:
            polarX = int((r * np.cos(t)) + img.shape[1] / 2)
            polarY = int((r * np.sin(t)) + img.shape[0] / 2)
            try:
                cartisian_image[r][int(m(t) - 1)] = img[polarY][polarX]
            except:
                pass

    return cartisian_image.astype("uint8")

# Function to remove reflection using inpainting
def remove_reflection(img):
    # Ensure the image is grayscale (single channel)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Generate binary mask using threshold
    ret, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # The mask should be single-channel (already binary), ensure it's 8-bit
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=2)

    # Inpainting using the corrected mask
    dst = cv2.inpaint(img, dilation, 5, cv2.INPAINT_TELEA)
    return dst

# Function to process the image and detect circles
def processing(image_path, r):
    success = False
    original_image = cv2.imread(image_path)
    display_image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), "Original Image")

    image = cv2.resize(original_image, (640, 480), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray, "Grayscale Image")

    gray_blurred = cv2.medianBlur(gray, 11)
    display_image(gray_blurred, "Blurred Image")

    ret, _ = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect circles (iris/pupil) using Hough Circles
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
            display_image(image_roi, "Detected Iris/Pupil")
        success = True
        return image_roi, radius, success
    else:
        print(f"{image_path} -> No circles (iris) found.")
        return None, None, success

# Function to display the rubber sheet model using Plotly
def display_rubber_sheet_plotly(rubber_sheet):
    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(z=rubber_sheet[..., 0], colorscale='gray'))

    # Set titles and labels
    fig.update_layout(
        title="Rubber Sheet Model",
        xaxis_title="Theta",
        yaxis_title="Radius",
        height=800,
        width=800
    )

    # Show the Plotly figure with zoom and interaction enabled
    fig.show()

# Main processing function
def process_img(image_path, keep_reflection=False):
    image_roi, rr, success = processing(image_path, 50)

    if success:
        if not keep_reflection:
            image_roi = remove_reflection(image_roi)
            display_image(image_roi, "After Reflection Removal")

        # Generate and display the rubber sheet model
        rubber_sheet = generate_rubber_sheet_model(image_roi)
        display_rubber_sheet_plotly(rubber_sheet)
    else:
        print("Iris or Pupil not detected.")

# Test the function on your image
image_path = 'eye.jpg'  # Replace with your image file
process_img(image_path)
