import cv2
import numpy as np


def translate(img, x, y):
    '''
    translate the image by (x, y)
    :param img: the original image
    :param x: the magnitude of the translation on the X-axis
    :param y: the magnitude of the translation on the Y-axis
    :return: the translated image
    '''
    height = img.shape[0]
    width = img.shape[1]
    M = np.float32([[1, 0, x], [0, 1, y]])
    img = cv2.warpAffine(img, M, (width, height))
    return img