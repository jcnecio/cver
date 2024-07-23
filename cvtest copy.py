import cv2 as cv
import numpy as np

from predict import predict

# load the games image
img = cv.imread("image.png")
	
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite("grey.png", img)

kernel = np.ones((2,2),np.float32)
img = cv.filter2D(img,-1,kernel)
cv.imwrite("filtered.png", img)

# high_thresh, thresh_im = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# lowThresh = 0.5*high_thresh

# img = cv.Canny(img, high_thresh, lowThresh)
# cv.imwrite("edges.png", img)

# img = cv.dilate(img, np.ones((2,2),np.float32))
# cv.imwrite("dilated.png", img)

# img = cv.erode(img, np.ones((2,2),np.float32))
# cv.imwrite("erroded.png", img)

# img = cv.Canny(img, lowThresh, high_thresh)
# cv.imwrite("edges2.png", img)

# img = cv.dilate(img, np.ones((2,2),np.float32))
# cv.imwrite("dilated2.png", img)

# img = cv.Canny(img, lowThresh, high_thresh)
# cv.imwrite("edges3.png", img)

# ret, thresh = cv.threshold(img, 127, 255, 0)
ret, thresh = cv.threshold(img, 127, 175, cv.THRESH_BINARY)
# thresh= cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
complete_contours = []
for contour in contours:
    if len(contour) > 20:
        complete_contours.append(contour)
pad = 3

found_images = []
for i, complete in enumerate(complete_contours):
    minx, miny = img.shape
    maxx, maxy = 0, 0
    for point in complete:
        x,y = point[0]
        if x > maxx:
            maxx = x
        if x < minx:
            minx = x
        if y > maxy:
            maxy = y
        if y < miny:
            miny = y
    contour_img = np.zeros((img.shape[0],img.shape[1], 1), dtype=np.uint8)
    cv.drawContours(contour_img, [complete], 0, (255,255,255), thickness=-1)

    area = (maxx-minx)*(maxy-miny)
    print(area)
    if area > 2000:
        halfy = int((maxy-miny)/2)
        halfx = int((maxx-minx)/2)
        detimg = contour_img[miny:maxy+pad, minx:minx+halfx+pad]
        found_images.append({
            'left': minx,
            'img': detimg
        })
        cv.imwrite(f"{i}_0contours.png", detimg)
        detimg = contour_img[miny:maxy+pad, minx+halfx:maxx+pad]
        found_images.append({
            'left': minx+halfx,
            'img': detimg
        })
        cv.imwrite(f"{i}_1contours.png", detimg)
    elif area < 500:
        pass
    else:
        detimg = contour_img[miny:maxy+pad, minx:maxx+pad]
        found_images.append({
            'left': minx,
            'img': detimg
        })
        # cv.imwrite(f"{i}contours.png", detimg)

found_images.sort(key=lambda x:x['left'])
images = []
for idx, image in enumerate(found_images):
    images.append(image['img'])
    cv.imwrite(f"{idx}_img.png", image['img'])
predict(images)

# for idx,complete_contour in enumerate(complete_contours):
contour_img = np.zeros((img.shape[0],img.shape[1], 1), dtype=np.uint8)
cv.drawContours(contour_img, complete_contours, -1, (255,255,255), thickness=-1)
cv.imwrite(f"contours.png", contour_img)