import cv2
import numpy as np



def getGrayHand(img):
    lower = np.array([0, 133, 77], np.uint8)
    upper = np.array([255, 173, 127], np.uint8)
    imageYcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegion = cv2.inRange(imageYcr, lower, upper)
    res = cv2.bitwise_and(imageYcr, imageYcr, mask=skinRegion)
    cv2.cvtColor(res, cv2.COLOR_YCrCb2BGR)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, final = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return final
# print(skinRegion)


#function to remove extra blackspace
def reSpace(img):
    minx = 1000
    miny = 1000
    maxx = 0
    maxy= 0
    for y,row in enumerate(img):
        for x,grayValue in enumerate(row):
            if(grayValue > 10):
                if(x < minx):
                    minx = x
                elif(x > maxx):
                    maxx = x
                if(y < miny):
                    miny = y
                elif(y > maxy):
                    maxy = y
    return img[miny:maxy, minx:maxx]

def getfinalImg(img):
    ret = getGrayHand(img)
    ret = reSpace(ret)
    return ret

def drawContourBin(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) > 0):
        contours = max(contours, key= lambda x:cv2.contourArea(x))
        print(contours)
    return img


def drawSkin(img):
    lower = np.array([0, 133, 77], np.uint8)
    upper = np.array([255, 173, 127], np.uint8)
    imageYcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegion = cv2.inRange(imageYcr, lower, upper)
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(img, contours, i, (0, 255, 0), 3)
    return img

def preprocess(action_frame):

    blur = cv2.GaussianBlur(action_frame, (3,3), 0)
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
    if( len(contours) != 0):
        c = max(contours, key=cv2.contourArea)
        if(cv2.contourArea(c) > 100):
            x, y, w, h = cv2.boundingRect(c)
            #print([x,y,w,h])
            return hsv_d[y:y+h,x:x+w]
    #hsv_d = reNoise(hsv_d)
    return hsv_d



