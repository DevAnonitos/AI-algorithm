import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgClass = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img2.jpg")
imgConvert = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img3.jpg")

kernel = np.ones((5,5), np.float32)/25
dst = cv.filter2D(imgClass, -1, kernel)

blurImage = cv.bilateralFilter(imgConvert, 9, 115, 115)

cv.imshow("test", dst)
cv.imshow("test", blurImage)
cv.waitKey(0)
cv.destroyAllWindows()
