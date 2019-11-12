import cv2
import numpy as np


def canny_filter(iframe):
    gray = cv2.cvtColor(iframe, cv2.COLOR_BGR2GRAY)

    # get contours
    edges = cv2.Canny(gray, 20, 30)
    edges_high_thresh = cv2.Canny(gray, 70, 120)

    #  for comparison
    images = np.hstack((gray, edges, edges_high_thresh))
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', images)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 255, 0], 2)
    return line_img


def bilateral_filtering(iframe):

    region_of_interest_vertices = [
        (0, iframe.shape[0]),
        (iframe.shape[1]/2, iframe.shape[0]/5),
        (iframe.shape[1], iframe.shape[0]),
    ]

    crop_iframe = crop_roi(iframe, np.array([region_of_interest_vertices], np.int32))
    gray = cv2.cvtColor(crop_iframe, cv2.COLOR_BGR2GRAY)

    # smoothing
    gray_filtered = cv2.bilateralFilter(gray, 7, 40, 40)

    # canny filter
    edges_high_thresh = cv2.Canny(gray_filtered, 100, 200)
    kernelS = 5
    guas_image = cv2.GaussianBlur(edges_high_thresh, (kernelS, kernelS), 0.5)

    rho = 7
    theta = np.pi/60
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 10
    min_line_len = 60

    max_line_gap = 20
    line_image = hough_lines(guas_image, rho, theta, threshold, min_line_len, max_line_gap)
    weighted = cv2.addWeighted(line_image, 1, iframe, 1, 0)

    images = np.hstack((guas_image, edges_high_thresh))
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', weighted)

def crop_roi(iframe, vertices):
    mask = np.zeros_like(iframe)
    channel_count = iframe.shape[2]

    match_mask_color = (255, ) *channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(iframe, mask)
    return masked_image


def mask_green(iframe):
    gray = cv2.cvtColor(iframe, cv2.COLOR_BGR2GRAY)

    # smoothing
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

    hsv = cv2.cvtColor(iframe, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([40,40,40], dtype = "uint8")
    # upper_green = np.array([70,255,255], dtype = "uint8")
    #
    # mask_greenimg = cv2.inRange(hsv, lower_green, upper_green)
    # mask_gr_image = cv2.bitwise_and(iframe, iframe, mask=mask_greenimg)

    mask_white = cv2.inRange(gray, 200, 255)
    mask_wh_image = cv2.bitwise_and(gray, gray, mask=mask_white)
    # gaussian
    kernelS = 3
    guas_image = cv2.GaussianBlur(mask_wh_image, (kernelS, kernelS), cv2.BORDER_DEFAULT)
    #canny
    canny_gaus = cv2.Canny(guas_image, 60, 120)

    images = np.hstack((gray, canny_gaus, guas_image))
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', images)



def imageFilt():
    img = cv2.imread('pic.jpg')
    bilateral_filtering(img)


def videoFilt():
    cap = cv2.VideoCapture('longwalk.mp4')

    if cap.isOpened() is False:
        print("Error")

    #  Read the video

    while cap.isOpened():

        ret, frame = cap.read()

        if ret is True:
            bilateral_filtering(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()

    cv2.destroyAllWindows()


videoFilt()
#while cv2.waitKey(1000) & 0xFF != ord('q'):
#    imageFilt()
#cv2.destroyAllWindows()