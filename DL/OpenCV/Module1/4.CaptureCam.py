import cv2

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/video/output.avi', fourcc, 20.0, (640, 480))

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Set the duration of the video to 10 seconds
duration = 10  # seconds

# Get the start time of the video
start_time = cv2.getTickCount()

# Start a loop to continuously capture frames from the camera and write them to the file
while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Write the frame to the file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects and destroy all windows
cap.release()
out.release()
cv2.destroyAllWindows()
