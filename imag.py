import cv2
import matplotlib.pyplot as plt
import numpy as np
cv2.namedWindow("hello")
img = cv2.imread("myHand.jpg")
print(img.shape)
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
x = 400
while(x < 500):
    y=100
    while(y < 200):
        resized[x,y] = [0,255,0]
        y += 1
    x += 1

print('Resized Dimensions : ', resized.shape)
print(resized.shape[0])






min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([255, 173, 127], np.uint8)
imageYCrCb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB)
skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
# Do contour detection on skin region
contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
armDist = []
c = 0
print(type(contours[0]))
print(contours[0][0][0][0])
print(type(contours[0][0]))

while ( c < len(contours[0])):
    for xy in contours[0]:
        # check if the y values in the array match up
        '''if(xy[0] == contours[c][0] and xy[1] != contours[c][1]):
            print(abs(xy[1]-contours[c][1]))'''
    c += 1


#print(contours)

# Draw the contour on the source image
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(resized, contours, i, (0, 255, 0), 3)

'''cv2.imshow("hello", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
px = img[400,400]
x = 400
'''while(x < 500):
    y = 400
    while(y < 500):
        img[x,y] = [0,0,255]
        y += 1
    x += 1'''
#cv2.imshow("hello", img)
#cv2.destroyWindow('hello')
#videoFrame.release()
