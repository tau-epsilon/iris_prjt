- [ ]   Work in progress.
# iris_prjt
Development of iris and pupil detection (and texture analysis) 🖥️

🔲 Reading and displaying images using cv2 :
- First install using pip (or in mamba), use `$ pip install cv2`
```python
import cv2
image_path = 'path/to/image'
image = cv2.imread(image_path) # read the image
cv2.imshow("Name of window", image)
cv2.waitKey(0) # Wait for any key to be pressed
cv2.destroyWindows()
```

* Iris segmentation with kmeans
  - `Kmeans` is method to find `clusters` in image
      - Here, `kmeans++` is used (from `scikit-learn`)
      - [X] Failure
* Iris detection from a face using mediapipe
* Four corner method for pupil or iris
* Daugman's algorithm
* Hough transform
* Differential methods (sobel, prewitt etc.)
* TODO Gradient descent
* TODO Four corner with gradient descent
