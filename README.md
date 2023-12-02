# Vision based People Tracking for Ubiquitous Augmented Reality Applications

import cv2
from google.colab import files
from IPython.display import display, Image

# Upload a sample video file
uploaded = files.upload()

# Get the file name of the uploaded video
video_file = next(iter(uploaded))

# Create a VideoCapture object using the uploaded video
video_capture = cv2.VideoCapture(video_file)

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')

while True:
    # Read the video frame
    ret, frame = video_capture.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    _, colab_frame = cv2.imencode('.jpeg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display(Image(data=colab_frame.tobytes()))

# Release the video capture object
video_capture.release()

