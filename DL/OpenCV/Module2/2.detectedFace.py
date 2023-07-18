import cv2
import mediapipe as mp

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

# Create a Mediapipe FaceDetection object
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)

while True:
    # Read a frame from the video stream
    success, img = cap.read()

    # Convert the image to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use the FaceDetection object to detect faces in the image
    results = faceDetection.process(imgRGB)
    if results.detections:
        for detection in results.detections:
            # Get the bounding box of the face
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            # Draw a rectangle around the face
            cv2.rectangle(img, bbox, (0, 255, 0), 2)

    # Display the image with the detected faces
    cv2.imshow("Face Detection", img)

    # Wait for a key press and exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
