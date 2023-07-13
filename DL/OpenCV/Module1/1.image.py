import cv2
import os
import numpy
from matplotlib import pyplot as plt

img = cv2.imread('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img2.jpg")

blur = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)

# test color chart
# plt.hist(image)
# plt.show()

# ret, thresh1 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

cv2.imshow("test", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# if img is None:
#     print('Can not read this file')
# else:
#     cv2.imshow('Image', img)
#     cv2.imwrite('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.png', img)

#     cv2.imshow("GrayScaleImage", gray_img)
#     cv2.waitKey(0)

#     cv2.destroyAllWindows()

# # Handle RandomByte
# randomByteArr = bytearray(os.urandom(300000))
# flatNumpyArr = numpy.array(randomByteArr)

# grayImage = flatNumpyArr.reshape(500, 600)
# blurImage = cv2.GaussianBlur(grayImage, (5,5), 0)


# cv2.imwrite('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/imgConvert.png', grayImage)

# cv2.imwrite('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/blur_gray_image.png', blurImage)
