import cv2
import numpy as np
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
try:
    model = load_model(r"keras_model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the labels
try:
    class_names = open(r"labels.txt", "r").readlines()
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

# Initialize the camera (0 is default camera)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open video device.")
    exit()

# Variables to keep track of previous prediction
prev_class = None

while True:
    # Capture frame from the camera
    ret, frame = camera.read()

    # If frame capture fails, skip the loop
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Resize the image to match the model input (224x224 pixels)
    image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert the image to a numpy array and reshape for the model input
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image to the range [-1, 1]
    image_array = (image_array / 127.5) - 1

    # Make predictions using the model
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Check if the prediction has changed
    if prev_class != class_name:
        prev_class = class_name

        # Clear the previous text on the frame
        output_frame = frame.copy()

        # Display the predicted class and confidence score on the frame
        text = f"Class: {class_name}, Confidence: {confidence_score * 100:.2f}%"
        cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the frame with prediction on the screen
        cv2.imshow("Webcam Image", output_frame)

    # Listen for keyboard input
    key = cv2.waitKey(1) & 0xFF
    # Exit the loop when 'Esc' is pressed
    if key == 27:  # ESC key
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()