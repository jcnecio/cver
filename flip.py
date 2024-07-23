import cv2 as cv

img = cv.imread("partial_contour.png")
cv.imwrite("flipped.png", cv.bitwise_not(img))