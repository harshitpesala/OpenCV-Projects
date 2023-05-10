import numpy as np
import cv2 as cv
from collections import deque
import imutils

cap = cv.VideoCapture('tennisVid.mp4')

def nothing(x):
    pass
"""
cv.namedWindow('Tracking')
cv.createTrackbar('LH', 'Tracking', 0 , 255, nothing)
cv.createTrackbar('LS', 'Tracking', 0 , 255, nothing)
cv.createTrackbar('LV', 'Tracking', 0 , 255, nothing)
cv.createTrackbar('UH', 'Tracking', 255 , 255, nothing)
cv.createTrackbar('US', 'Tracking', 255 , 255, nothing)
cv.createTrackbar('UV', 'Tracking', 255 , 255, nothing)
"""

pts = deque(maxlen=1000)

l_h = 29
l_s = 40
l_v = 6

u_h = 95
u_s = 255
u_v = 255


while cap.isOpened():
    ret, frame = cap.read()
    frame=frame[170:800,320:1430]

    if not ret:
        print(" Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(frame, (11,11), 0)

    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)


    """l_h = cv.getTrackbarPos("LH", "Tracking")
    l_s = cv.getTrackbarPos("LS", "Tracking")
    l_v = cv.getTrackbarPos("LV", "Tracking")

    u_h = cv.getTrackbarPos("UH", "Tracking")
    u_s = cv.getTrackbarPos("US", "Tracking")
    u_v = cv.getTrackbarPos("UV", "Tracking")"""

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv.inRange(hsv, l_b, u_b)
    mask = cv.dilate(mask, None, iterations=1)

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        # finding the largest contour in the mask and then using it to compute min enclosing area circle and centroid
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 0:
            # draw the circle and centroid on the frame
            # updating the list of tracked points
            cv.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv.circle(frame, center, 5, (0, 0, 255), -1)
    

    #res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    #cv.imshow('res', res)
    #cv.imshow('blur', blur)
    if cv.waitKey(3) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()