import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("temp.png")
cv2.imshow("image", image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
blurred = cv2.GaussianBlur(gray, (21, 21), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
plt.hist(gray.ravel(), 256, [0, 256])
plt.show()

thresh = cv2.threshold(gray, 220, 230, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)
