import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import *
from sklearn.linear_model import LinearRegression

points = []


def read_image(name):
    img = cv2.imread(name)

    return img


def greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, lt, ht):
    return cv2.Canny(img, lt, ht)


def gaussBlur(img, k):
    return cv2.GaussianBlur(img, (k,k), 0)


def roiIm(img, vertices):

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        match_mask_color = (255, ) * channel_count
    else:
        match_mask_color = 255

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    indices = []

    for i, line in enumerate(lines):
        theta = atan(abs(line[0][3] - line[0][1])/abs(line[0][2] - line[0][0]))
        if (theta == 0 or theta == radians(180) or (radians(45) < theta < radians(135))):
            indices.append(i)

    lines = np.delete(lines, indices, 0)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 255, 0], 2)
    return line_img


def proc_image(iframe):
    grey = greyscale(iframe)

    k = 7
    blurG = gaussBlur(grey, k)
    blurG = gaussBlur(blurG, k)
    ost = otsuBin(iframe)
    lt = 0
    ht = 255
    edge = canny(ost, lt, ht)
    cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
    cv2.imshow("edge", edge)
    h = iframe.shape[0]
    w = iframe.shape[1]

    roi_vertices = [
        (0, h), (0, h/3),
        (w, h / 3),
        (w, h)
    ]
    roi_vertices = np.int32([roi_vertices])
    roi = roiIm(edge, roi_vertices)
    test = roiIm(iframe, roi_vertices)
    cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
    cv2.imshow("roi", test)
    rho = 2
    theta = np.pi/180
    threshold = 30
    min_line_length = 30
    max_line_gap = 10
    line_imag = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)

    res = cv2.addWeighted(line_imag, 1.0,  iframe, 1.0, 0.0)

    return res


def mask_image(iframe):
    hsv = cv2.cvtColor(iframe, cv2.COLOR_BGR2HSV)
    ret, thresh = cv2.threshold(hsv, 240, 255, cv2.THRESH_BINARY)
    iframe[thresh == 255] = 0
    lower_gray = np.array([3, 4, 21], dtype="uint8")
    upper_gray = np.array([30, 50, 140], dtype="uint8")

    mask_gr = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_gr_img = cv2.bitwise_and(iframe, iframe, mask=mask_gr)
    #cv2.namedWindow('final', cv2.WINDOW_NORMAL)
    #cv2.imshow("final", iframe)

    #res = proc_image(iframe)
    return mask_gr_img



def video():
    cap = cv2.VideoCapture("longwalk.mp4")
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is True:
            res = findEdge(frame)
            cv2.namedWindow('final', cv2.WINDOW_NORMAL)
            cv2.imshow("final", res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def image():
    img = read_image("pic.jpg")
    res = otsuBin(img)
    #res = mask_image(img)
    while cv2.waitKey(1000) & 0xFF != ord('q'):
        cv2.namedWindow('final', cv2.WINDOW_NORMAL)
        cv2.imshow("final", res)


def threshold(img):
    #img = read_image("pic.jpg")
    #img = greyscale(img)
    #img = gaussBlur(img, 5)
    ret, thresh1 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 130, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 130, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 130, 255, cv2.THRESH_TOZERO_INV)
    # edge = canny(thresh2, 130, 150)
    # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    # images = [img, thresh1, thresh2, thresh3, thresh4, edge]
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    return thresh1


def otsuBin(img):
    #img = read_image("pic.jpg")
    img = greyscale(img)
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 10)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #th3 = ~th3
    # plot all the images and their histograms
    # images = [img, 0, th1,
    #           img, 0, th2,
    #           blur, 0, th3]
    # titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
    #           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
    #           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    # for i in range(3):
    #     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    #     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    #     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    #     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    #plt.show()

    return th3



def drawLine(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2,y2), [255, 255, 0], 8)

def findEdge(img):
    #img = read_image("icegrass1.jpg")
    #masked = mask_image(img)
    ost = otsuBin(img)
    h, w, r = img.shape
    lt = 0
    ht = 255
    edge = canny(ost, lt, ht)
    # return edge
    # edge = crop(edge)
    # Left
    Left = crop(edge, 1)
    Right = crop(edge, 0)
    indicesLeft = np.where(Left == 255)
    indicesRight = np.where(Right == 255)
    yLeft = indicesLeft[0]
    xLeft = indicesLeft[1]
    yRight = indicesRight[0]
    xRight = indicesRight[1]
    Lx1, Ly1, Lx2, Ly2 = linearFit(xLeft, yLeft)
    drawLine(img, Lx1, Ly1, Lx2, Ly2)
    Rx1, Ry1, Rx2, Ry2 = linearFit(xRight, yRight)
    drawLine(img, Rx1, Ry1, Rx2, Ry2)
    drawLine(img, (Rx1+Lx1)/2, (Ry1+Ly1)/2, (Rx2+Lx2)/2, (Ry2 + Ly2)/2)
    return img


def linearFit(x,y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    y_pred = model.predict(x)
    return x[0], y[0], x[-1], y_pred[-1]


def crop(im, left):
    h, w = im.shape

    if left == 1:

        roi_vertices = [
            (w/2.2, h / 2), (w/2.2, h/5), (3*w / 4, h/5),
            (w, h / 2), (w, h), (3*w / 4, h)
        ]
        roi_vertices = [
            (w / 2, h), (w / 2, h / 4),
            (w, h / 4),
            (w, h)
        ]
    else:

        roi_vertices = [
            (0, h / 2), (0, h), (w / 4, h),
            (w / 1.8, h / 2), (w / 1.8, h/5), (w / 4, h/5)
        ]

        roi_vertices = [
            (0, h), (0, h / 4),
            (w/2, h / 4),
            (w/2, h)
        ]
    roi_vertices = np.int32([roi_vertices])
    roi = roiIm(im, roi_vertices)
    return roi

video()