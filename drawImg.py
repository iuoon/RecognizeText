import cv2
import numpy as np

if __name__ == "__main__":
    newImageInfo = (500, 500, 3)
    dst = np.zeros(newImageInfo, np.uint8)
    # 绘制扇形  1.目标图片  2.椭圆圆心    4.偏转角度  5.圆弧起始角度  6.终止角度  7.颜色
    cv2.ellipse(dst, (256, 256),(150,100) ,0,  145, (255, 255, 0))
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
