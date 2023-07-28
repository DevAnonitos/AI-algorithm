import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img6.jpg', cv.IMREAD_GRAYSCALE)

imgBlur = cv.medianBlur(img, 5)

assert img is not None, "file could not be read, check with os.path.exists()"
# ret, thresh1 = cv.threshold(img, 125, 255, cv.THRESH_BINARY)
# ret, thresh2 = cv.threshold(img, 125, 255, cv.THRESH_BINARY_INV)
# ret, thresh3 = cv.threshold(img, 125, 255, cv.THRESH_TRUNC)
# ret, thresh4 = cv.threshold(img, 125, 255, cv.THRESH_TOZERO)
# ret, thresh5= cv.threshold(img, 125, 255, cv.THRESH_TOZERO_INV)


# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()

ret, th1 = cv.threshold(imgBlur, 125, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [imgBlur, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()



