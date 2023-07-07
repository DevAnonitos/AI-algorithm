import cv2

img = cv2.imread('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if img is None:
    print('Can not read this file')
else:
    cv2.imshow('Image', img)
    cv2.imwrite('D:/2-bai tap cua dev/PythonAI/AI/DL/OpenCV/Module1/img/img1.png', img)

    cv2.imshow("GrayScaleImage", gray_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
