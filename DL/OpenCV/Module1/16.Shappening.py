import cv2 as cv
import numpy as np

image = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg")
cv.imshow('Original', image)

kernel_1 = np.array([
    [1,1,1],
    [1,-8,1],
    [1,1,1],
])

blur = cv.GaussianBlur(image, (5,5), cv.BORDER_DEFAULT)

output_1 = cv.filter2D(blur, -1, kernel_1)

cv.imshow("Output", output_1)
cv.waitKey(0)
cv.destroyAllWindows()
