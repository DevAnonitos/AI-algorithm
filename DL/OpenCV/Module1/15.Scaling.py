import numpy as np
import cv2 as cv

image = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg")
assert image is not  None, "file could not be read, check with os.path.exists()"

res = cv.resize(image,None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

cv.imshow("test", res)
cv.waitKey(0)
cv.destroyAllWindows()
