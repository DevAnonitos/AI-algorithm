import cv2 as cv
import numpy as np

img = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg", cv.IMREAD_GRAYSCALE)

# resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)

cropped = img[15:550, 35:500]


blur= cv.GaussianBlur(resized, (5,5), cv.BORDER_DEFAULT)

# sobel_h = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=5)
# sobel_v = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=5)

# cv.imshow('Original', img)
# cv.imshow('Sobel_h', sobel_h)
# cv.imshow('Sobel_v', sobel_v)
cv.imshow("Cropped", cropped)
cv.waitKey(0)
cv.destroyAllWindows()
