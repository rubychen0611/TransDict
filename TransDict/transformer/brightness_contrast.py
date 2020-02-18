import cv2
import numpy as np

def contrast(img, alpha):
    '''
    Adjust image contrast
    :param img: the original image
    :param alpha: [0,3]
    :return: the adjusted image
    '''
    img = np.uint8(np.clip(alpha * np.uint16(img), 0, 255))
    return img

def brightness(img, beta):
    '''
    Adjust image brightness
    :param img: the original image
    :param beta: [-100,100]
    :return: the adjusted image
    '''
    img = np.uint8(np.clip(np.uint16(img)+beta, 0, 255))
    return img