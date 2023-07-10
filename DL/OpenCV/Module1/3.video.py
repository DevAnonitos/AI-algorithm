import cv2

# Thiết lập các thông số cho file video
new_width, new_height = 880, 550
fps = 144
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/video/NewVideo1.mp4', fourcc, fps, (new_width, new_height))

# Đọc file video
cap = cv2.VideoCapture('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/video/Video1.mp4')

# Lặp qua từng frame của video
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    out.write(resized_frame)

    cv2.imshow('frame', resized_frame)

    if cv2.waitKey(40) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

