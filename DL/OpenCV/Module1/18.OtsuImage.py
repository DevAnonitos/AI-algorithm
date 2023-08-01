import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg", cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"

# Thresh Bin
ret1, thresh1 = cv.threshold(img, 125, 255, cv.THRESH_BINARY)

# Thresh with Otsu
ret2, thresh2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Thresh Otsu with Gaussian blur
blur = cv.GaussianBlur(img, (5,5), -1)
ret3, thresh3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

images = [
    img, 0, thresh1,
    img, 0, thresh2,
    blur, 0, thresh3
]

titles = [
    'Original Noisy Image','Histogram',
    'Global Thresholding (v=155)',
    'Original Noisy Image',
    'Histogram',
    "Otsu's Thresholding",
    'Gaussian filtered Image',
    'Histogram',
    "Otsu's Thresholding"
]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
