import cv2 as cv
import numpy as np

image = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg")
assert image is not None, "file could not be read, check with os.path.exists()"

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imageGray, 155, 185, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour =cv.drawContours(image, contours, -1, (0,255,0), 3)

cv.imshow("test", contour)
cv.waitKey(0)
cv.destroyAllWindows()
