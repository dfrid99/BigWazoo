import cv2
import numpy as np


def preprocess(action_frame):
    blur = cv2.GaussianBlur(action_frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    hsv_d = cv2.erode(blur, kernel)
    hsv_d = cv2.dilate(hsv_d, kernel)

    ret, thresh = cv2.threshold(hsv_d, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) != 0):
        c = max(contours, key=cv2.contourArea)
        if (cv2.contourArea(c) > 100):
            x, y, w, h = cv2.boundingRect(c)
            # print([x,y,w,h])
            return hsv_d[y:y + h, x:x + w]
    # hsv_d = reNoise(hsv_d)
    return hsv_d
