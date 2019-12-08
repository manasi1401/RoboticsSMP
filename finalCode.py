import cv2
import numpy as np
from math import *


#  Mask the side walk
def mask_image(iframe):
    hsv = cv2.cvtColor(iframe, cv2.COLOR_BGR2HSV)
    ret, thresh = cv2.threshold(hsv, 240, 255, cv2.THRESH_BINARY)
    iframe[thresh == 255] = 0
    #lower_gray = np.array([3, 4, 21], dtype="uint8") #allgrass, allsnow
    lower_gray = np.array([0, 0, 0], dtype="uint8") # 1x, 1.5x
    #upper_gray = np.array([30, 40, 220], dtype="uint8")
    #upper_gray = np.array([200, 30, 220], dtype="uint8") #1x
    upper_gray = np.array([220, 80, 230], dtype="uint8") #1.5x, 1x
    #upper_gray = np.array([60, 50, 240], dtype="uint8")
    #upper_gray = np.array([40, 60, 200], dtype="uint8") #allgrass, allsnow

    mask_gr = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_gr_img = cv2.bitwise_and(iframe, iframe, mask=mask_gr)

    return mask_gr_img


#  Otsus Binarization for thresholding
def otsuBin(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    if isnan(avgX) is False:
        cv2.line(masked, (int(w/2), int(h)), (int(avgX), y), [255, 255, 0], 8)
        return masked, avgX, y, h, w
    else:
        return masked, h/2, w/2, h, w


#  Capture the Video
def video():
    #cap = cv2.VideoCapture("2x_noshadows.avi")
    #cap = cv2.VideoCapture("1x_minorshadows.avi")
    #cap = cv2.VideoCapture("1.5x_noshadows.avi")
    cap = cv2.VideoCapture("Sunday.mp4")
    #cap = cv2.VideoCapture("allsnow.mp4")
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is True:
            res, x, y, h, w = averageMidPoint(frame, 200)
            print(x, y, h, w)
            #print(frame[int(y)][int(x)][:])
            cv2.namedWindow('final', cv2.WINDOW_NORMAL)
            cv2.imshow("final", res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()

video()