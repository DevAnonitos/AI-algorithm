import cv2 as cv
import numpy as np
# doc hinh anh vao tu thu muc
img = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img4.png", cv.IMREAD_GRAYSCALE)
# chay vong lap neu khong doc dc tra ve loi
assert img is not None, "file could not be read, check with os.path.exists()"
# kich thuong kernel ma tran 5x5
kernel = np.ones((5,5), np.uint8)
# lam xoi mon(hep hinh anh)
erosion = cv.erode(img, kernel, iterations=1)
# lam gian noi hinh anh
dilation = cv.dilate(img, kernel, iterations=1)
# lam do doc hinh thai cua hinh anh
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# show ra hinh anh sau bien doi
cv.imshow("test", erosion)
cv.imshow("test1", dilation)
cv.imshow("test2", gradient)
cv.waitKey(0)
cv.destroyAllWindows()
