import numpy as np
import cv2
'''
================================
test4:提取二维码
================================
'''

code_name = "C:\\Users\\iuoon\\Desktop\\test\\temp\\cropped_img.jpg"
image = cv2.imread(code_name)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
#gradient = cv2.GaussianBlur(gray,(5,5),1.5,None,1.5)
cv2.imshow("gradient",gradient)
#原本没有过滤颜色通道的时候，这个高斯模糊有效，但是如果进行了颜色过滤，不用高斯模糊效果更好
#blurred = cv2.blur(gradient, (11, 11))
(_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh)
#cv2.imwrite('thresh.jpg',thresh)
qrRatio = 15
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed",closed)
#cv2.imwrite('closed.jpg',closed)

closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
#cv2.imwrite('closed1.jpg',closed)

cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
for i in  range(len(cnts)):
    cnt = cnts[i]
    area = cv2.contourArea(cnt)
    # 面积小的都筛选掉、这个1000可以按照效果自行设置
    if(area < 1000):
        continue
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)


# x, y, w, h = cv2.boundingRect(c)
# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('closed1.jpg',image)
cv2.imshow("Image", image)

cv2.waitKey(0)
