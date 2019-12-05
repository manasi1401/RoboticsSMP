import cv2
import numpy as np
from math import *

#  Read the image file
def read_image(name):
    img = cv2.imread(name)
    return img


#  Return greyscale image
def greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


#  Mask the side walk
def mask_image(iframe):
    hsv = cv2.cvtColor(iframe, cv2.COLOR_BGR2HSV)
    ret, thresh = cv2.threshold(hsv, 240, 255, cv2.THRESH_BINARY)
    iframe[thresh == 255] = 0
    lower_gray = np.array([3, 4, 21], dtype="uint8")
    upper_gray = np.array([40, 60, 200], dtype="uint8")

    mask_gr = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_gr_img = cv2.bitwise_and(iframe, iframe, mask=mask_gr)

    return mask_gr_img


#  Otsu’s Binarization for thresholding
def otsuBin(img):
    img = greyscale(img)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 10)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


#  Return the midpoint of the lane
#  Returns x(col), y(row), h(image height, row), r(image width, col)
def averageMidPoint(img, y):
    masked = mask_image(img)
    ost = otsuBin(masked)
    h, w, r = img.shape

    index = np.where(ost[y][:] == 255)
    x = index[0]
    avgX = np.average(x)
    drawLine(img, int(w/2), int(h), int(avgX), y)
    return img, avgX, y, h, r


# draw line on the image
def drawLine(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2,y2), [255, 255, 0], 8)


#  Capture the Video
def video():
    cap = cv2.VideoCapture("allgrass.mp4")
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is True:
            res, x, y, h, r = averageMidPoint(frame, 600)
            cv2.namedWindow('final', cv2.WINDOW_NORMAL)
            cv2.imshow("final", res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()

video()