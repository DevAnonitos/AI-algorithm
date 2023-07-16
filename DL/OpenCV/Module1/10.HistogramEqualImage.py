import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img5.jpg", cv.IMREAD_GRAYSCALE)

# EqualImage
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))

# CLAHE image
clahe = cv.createCLAHE(clipLimit=20.5, tileGridSize=(20,20))
cl1 = clahe.apply(img)

plt.hist(cl1.ravel(),256,[0,256]);
plt.show()
cv.imshow("test", cl1)
cv.waitKey(0)
cv.destroyAllWindows()
