import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("temp.png")
cv2.imshow("image", image)
cv2.waitKey(0)
blurred = cv2.GaussianBlur(image, (85, 85), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
cv2.waitKey(0)

lower = [18, 50, 50]
upper = [35, 255, 255]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")
mask = cv2.inRange(hsv, lower, upper)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


output = cv2.bitwise_and(image, hsv, mask=mask)
cv2.imshow("output", output)
cv2.waitKey(0)

img = image.copy()
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', img)
cv2.waitKey(0)
