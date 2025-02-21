# Sensemi Iris & Pupil Ratio Determination

This project focuses on determining the **iris to pupil ratio** and analyzing their textures using various computer vision and machine learning techniques. It includes methods like **Iris Segmentation, Iris Detection, and Advanced Image Processing** to achieve accurate measurements of the iris and pupil for biometric applications or eye-related studies.

## Features & Techniques Used

- **Iris Segmentation with K-means Clustering**
  - K-means is a clustering technique used to partition the image into distinct groups.
  - Here, the **K-means++** method from **scikit-learn** is used for better initialization.
  - *Status: [X] Failure* – This approach didn't yield optimal results for iris segmentation.

- **Iris Detection from a Face using Mediapipe**
  - Utilizes **Mediapipe** for real-time iris detection by detecting the face landmarks.
  - *Status: [X] Successful* – Accurate iris detection using Mediapipe's face mesh model.

- **Four-Corner Method for Pupil/Iris Detection**
  - Uses the four corner coordinates to define the iris or pupil region.
  - This method helps in estimating the position and size of the iris/pupil.

- **Daugman’s Algorithm**
  - A well-known algorithm for iris recognition, which uses circular Hough transform to model the iris boundary.
  - This method is essential in biometric systems for iris-based identification.

- **Hough Transform**
  - Applied to detect circular boundaries (for both iris and pupil), which are assumed to be circular in shape.
  - Effective in finding the center and radius of the iris or pupil.

- **Differential Methods (Sobel, Prewitt, etc.)**
  - These methods are used for edge detection to outline the iris and pupil boundaries.
  - They help in detecting the fine contours of the eye region for more precise measurements.

## Project Directories

- **programs**: Contains the code for implementing the various techniques discussed above.
- **presentations**: Contains the presentation materials related to the project.
- **report**: Includes the final report detailing the methodology, results, and analysis of the iris and pupil ratio determination.

## Iris and Pupil Ratio Details

The iris-to-pupil ratio is determined by analyzing the relative sizes of the iris and pupil regions in the eye. The ratio can be used in biometric systems for identification or authentication, as well as in medical research to study conditions like anisocoria (unequal pupil sizes). Various image processing techniques, like segmentation, edge detection, and contour finding, are used to precisely measure the size of the pupil and iris.

### Example:
- The diameter of the pupil and iris are extracted from the segmented image.
- The ratio is calculated by dividing the pupil diameter by the iris diameter.
