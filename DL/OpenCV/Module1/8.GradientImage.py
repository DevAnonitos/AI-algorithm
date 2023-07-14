import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg", cv.IMREAD_GRAYSCALE)
imageTest = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img5.jpg", cv.IMREAD_GRAYSCALE)

assert imageTest is not None, "file could not be read, check with os.path.exists()"

# TestCase1
blur = cv.GaussianBlur(image, (5,5), cv.BORDER_DEFAULT)
edge = cv.Canny(blur, 50, 150)
laplacian = cv.Laplacian(blur, cv.CV_64F)
sobelX = cv.Sobel(image, cv.CV_64F, 1,0, ksize=5)
sobelY = cv.Sobel(edge, cv.CV_64F, 0,1, ksize=5)

# TestCase2
blur = cv.GaussianBlur(imageTest, (5,5), cv.BORDER_DEFAULT)
edge = cv.Canny(blur, 175, 35)
sobelX8U = cv.Sobel(edge, cv.CV_8U,1,0, ksize=5)

sobelX164f = cv.Sobel(edge, cv.CV_64F, 1,0, ksize=5)
abs_sobel64f = np.absolute(sobelX164f)
sobel_8u = np.uint8(abs_sobel64f)

# ShowCase1

# cv.imshow("test", laplacian)
# cv.imshow("testCase", sobelX)
# cv.imshow("test", sobelY)

# ShowCase2

plt.subplot(1,3,1),plt.imshow(imageTest,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelX8U,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
