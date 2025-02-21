# Make sure to remove iris_pupil_data.csv if doing detection for a fresh candidate.
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# Remove the CSV file if it already exists
csv_file = 'iris_pupil_data.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)

# Initialize Mediapipe Face Mesh
success, image = cap.read()
if not success:
    print("Ignoring empty camera frame.")
    continue

# Flip and convert the image to RGB
image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
image.flags.writeable = False

# Process the image and detect face landmarks
results = face_mesh.process(image)

# Convert the image color back for rendering
image.flags.writeable = True
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Skip the first 50 frames and the last 50 frames
if process_start_frame <= frame_number < process_end_frame:
    # Initialize data to write for the current frame
    data_to_write = [frame_number, None, None, None, None, None, None, None, None, None, None, None, None]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Function to calculate the distance between two landmarks
            def calculate_distance(landmark1, landmark2):
                x1, y1 = face_landmarks.landmark[landmark1].x, face_landmarks.landmark[landmark1].y
                x2, y2 = face_landmarks.landmark[landmark2].x, face_landmarks.landmark[landmark2].y
                pixel_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * np.array([image.shape[1], image.shape[0]])
                return np.linalg.norm(pixel_distance)

            # Calculate the distance between the outer corners of the eyes
            eye_distance = calculate_distance(LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER)

            # Check if the person is within the 59-60 cm range
            if pixel_distance_min <= eye_distance <= pixel_distance_max:

                # Function to calculate center and radius for given indices
                def calculate_center_and_radius(indices):
                    x = np.mean([face_landmarks.landmark[i].x for i in indices])
                    y = np.mean([face_landmarks.landmark[i].y for i in indices])
                    center = (int(x * image.shape[1]), int(y * image.shape[0]))
                    radius = int(np.linalg.norm(np.array([
                        face_landmarks.landmark[indices[0]].x - face_landmarks.landmark[indices[2]].x,
                        face_landmarks.landmark[indices[0]].y - face_landmarks.landmark[indices[2]].y
                    ]) * np.array([image.shape[1], image.shape[0]])) / 2)
                    return center, radius

                # Calculate and draw circles for left iris
                left_iris_center, left_iris_radius = calculate_center_and_radius(LEFT_IRIS)
                cv2.circle(image, left_iris_center, left_iris_radius, (0, 255, 0), 2)  # Green color for iris
                data_to_write[1:4] = [left_iris_center[0], left_iris_center[1], left_iris_radius]

                # Calculate and draw circles for right iris
                right_iris_center, right_iris_radius = calculate_center_and_radius(RIGHT_IRIS)
                cv2.circle(image, right_iris_center, right_iris_radius, (0, 255, 0), 2)  # Green color for iris
                data_to_write[4:7] = [right_iris_center[0], right_iris_center[1], right_iris_radius]

                # Calculate and draw circles for left pupil
                left_pupil_center, left_pupil_radius = calculate_center_and_radius(LEFT_PUPIL)
                if left_pupil_radius > 0:  # Ensure valid radius before drawing
                    cv2.circle(image, left_pupil_center, left_pupil_radius, (255, 0, 0), 2)  # Blue color for pupil
                    data_to_write[7:10] = [left_pupil_center[0], left_pupil_center[1], left_pupil_radius]
                else:
                    # Set default values if the radius is not valid
                    data_to_write[7:10] = ['NA', 'NA', 'NA']

                    # Calculate and draw circles for right pupil
                    right_pupil_center, right_pupil_radius = calculate_center_and_radius(RIGHT_PUPIL)
                    if right_pupil_radius > 0:  # Ensure valid radius before drawing
                        cv2.circle(image, right_pupil_center, right_pupil_radius, (255, 0, 0), 2)  # Blue color for pupil
                        data_to_write[10:13] = [right_pupil_center[0], right_pupil_center[1], right_pupil_radius]
                    else:
                        # Set default values if the radius is not valid
                        data_to_write[10:13] = ['NA', 'NA', 'NA']

                        # Write data for the current frame to the CSV file
                        writer.writerow(data_to_write)

                    else:
                        # Indicate whether the person is too close or too far
                        if eye_distance < pixel_distance_min:
                            cv2.putText(image, "Move closer", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif eye_distance > pixel_distance_max:
                            cv2.putText(image, "Go back", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the image with detected iris and pupil
            cv2.imshow('Iris and Pupil Detection', image)
            frame_number += 1

            # Exit if 'Esc' is pressed
            if cv2.waitKey(5) & 0xFF == 27:  # 'Esc' key to exit
                break

    cap.release()
    cv2.destroyAllWindows()
