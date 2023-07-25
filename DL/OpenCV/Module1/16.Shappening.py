import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# image = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg")
# cv.imshow('Original', image)

# HistoGram
img = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img10.jpg", 0)

height = 800
width = 800

resizeImage = cv.resize(img, (height, width))

img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])

imgOutput = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

histoImage = cv.equalizeHist(resizeImage)

# Handle Shape
# kernel_1 = np.array([
#     [1,1,1],
#     [1,-8,1],
#     [1,1,1],
# ])

# blur = cv.GaussianBlur(image, (5,5), cv.BORDER_DEFAULT)

# output_1 = cv.filter2D(blur, -1, kernel_1)

cv.imshow("Output", resizeImage)
cv.imshow("ShowHist", histoImage)

cv.imshow("Color input image", resizeImage)
cv.imshow("image output", imgOutput)

cv.waitKey(0)
cv.destroyAllWindows()
