import cv2
import numpy as np

def USM_sharpen(img):
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    #dst = cv2.filter2D(img, -1, kernel=kernel)
    #return dst