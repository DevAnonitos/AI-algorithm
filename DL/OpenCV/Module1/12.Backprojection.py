import numpy as np
import cv2 as cv


roi = cv.imread("D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg")
assert roi is not None, "file could not be read, check with os.path.exists()"
hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)



# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsv],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(hsv,thresh)
res = np.vstack((hsv,thresh,res))

cv.imshow("test", res)
cv.waitKey(0)
cv.destroyAllWindows()
