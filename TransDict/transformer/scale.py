import cv2
import numpy as np
from TransDict.core.exceptions import TransformationError
def scale(img, ratio):
    '''
    scale the image with a specified ratio
    :param img: the original image
    :param ratio: scaling ratio (>0)
    :return: the scaled img
    '''
    height0 = img.shape[0]
    width0 = img.shape[1]
    #print(img.shape)
    if ratio <= 0:
        raise TransformationError('The ratio should be positive.')
    img1 = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    if ratio > 1:
        top = int((height1-height0)/2)
        left = int((width1-width0)/2)
        img = img1[top:top + height0, left:left + width0]
    elif ratio < 1:
        top = int((height0 - height1) / 2)
        left = int((width0 - width1) / 2)
        img = np.zeros(shape=(height0,width0,3)).astype(np.float32)
        img[top:top+height1, left:left+width1, :] = img1
        #print(img.shape)
    return img
