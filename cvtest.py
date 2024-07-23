import cv2 as cv
import numpy as np

from predict import predict

splits = [-10, -5, 0, 5, 10]
img = cv.imread("download.png")
	
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite("grey.png", img)

kernel = np.ones((2,2),np.float32)
img = cv.filter2D(img,-1,kernel)
cv.imwrite("filtered.png", img)

ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
complete_contours = []
for contour in contours:
    if len(contour) > 20:
        complete_contours.append(contour)
pad = 3

found_images = []
minx, miny = img.shape
maxx, maxy = 0, 0
for i, complete in enumerate(complete_contours):
    if cv.contourArea(complete) < 150:
        pass
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

blank_img = np.zeros((img.shape[0]+pad,img.shape[1]+pad, 1), dtype=np.uint8)
cv.drawContours(blank_img, complete_contours, -1, (255,255,255), -1)
blank_img = cv.erode(blank_img, np.ones((2,2)))

copy_blank = blank_img
partial_img = blank_img[miny:maxy, minx:maxx]
cv.imwrite("partial.png", partial_img)

ret, thresh = cv.threshold(blank_img, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# print(hierarchy)

blank_img = np.zeros((img.shape[0]+pad,img.shape[1]+pad, 1), dtype=np.uint8)
found_images = []

def split(minx,miny,maxx,maxy):
    area = (maxx-minx)*(maxy-miny)
    cv.drawContours(blank_img, [contour], -1, (255,255,255), thickness=2)
    if area > 1500:
        halfx = int((maxx-minx)/2)
        split(minx,miny,minx+halfx,maxy)
        split(minx+halfx,miny,maxx,maxy)
    elif area < 500:
        pass
    else:
        detimg = copy_blank[miny:maxy+pad, minx:maxx+pad]
        found_images.append({
            'left': minx,
            'img': detimg
        })

for idx, contour in enumerate(contours):
    if len(contour) > 20:
        minx, miny, _ = blank_img.shape
        maxx, maxy = 0, 0
        if hierarchy[0][idx][3] != -1:
            continue
        for point in contour:
            x,y = point[0]
            if x > maxx:
                maxx = x
            if x < minx:
                minx = x
            if y > maxy:
                maxy = y
            if y < miny:
                miny = y
        split(minx, miny, maxx, maxy)

# blank_img = np.zeros((img.shape[0]+pad,img.shape[1]+pad, 1), dtype=np.uint8)
# cv.drawContours(blank_img, contours, -1, (255,255,255), thickness=2)
cv.imwrite(f"partial_contour.png", blank_img)

# images = []
# partialX = partial_img.shape[1]
# for x in range(6):
#     for s in splits:
#         startX = max(0, int(x*(partialX/6))+s)
#         endX = min(partialX, int((x+1)*(partialX/6))+s)
#         images.append(partial_img[:,startX:endX])

found_images.sort(key=lambda x:x['left'])
# last_image = found_images[0]
# distinct_images = [last_image]

# if last_image:
#     for i in range(1,len(found_images)):
#         curr_image = found_images[i]
#         pts_a = cv.findNonZero(last_image['img'])
#         pts_b = cv.findNonZero(curr_image['img'])
#         hd = cv.createHausdorffDistanceExtractor()
#         result = hd.computeDistance(pts_a, pts_b)
#         print(result)
#         if result < 2.5:
#             continue
#         last_image = curr_image
#         distinct_images.append(last_image)

images = []
for idx, image in enumerate(found_images):
    images.append(image['img'])
    cv.imwrite(f"{idx}_img.png", image['img'])
predict(images)

contour_img = np.zeros((img.shape[0],img.shape[1], 1), dtype=np.uint8)
cv.drawContours(contour_img, complete_contours, -1, (255,255,255), thickness=-1)
cv.imwrite(f"contours.png", contour_img)