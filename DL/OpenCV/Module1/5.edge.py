import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

Img = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg", cv.IMREAD_ANYCOLOR)

# blur = cv.GaussianBlur(Img, (3,3), cv.BORDER_DEFAULT)
# edges = cv.Canny(blur, 50, 150)

cols, rows = Img.shape[0:2]
# print(h, w)

pts1 = np.float32([[211, 701],[221, 777], [292, 707], [287, 777]])
pts2 = np.float32([[0,0],[903,0],[0,486],[903,486]])

# plt.imshow(Img)
# plt.show()

M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(Img,M,(cols,rows))

# M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
# dst = cv.warpAffine(Img,M,(cols,rows))

cv.imshow("test", dst)
cv.waitKey(0)
cv.destroyAllWindows()
