import cv2
import numpy as np

def hue(img, dh):
    '''
        Adjust hue.
        :param img:
        :param ds: (-100, 100)
        :return:
        '''
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hlsImg = np.uint16(hlsImg)
    MAX_VALUE = 100
    hlsImg[:, :, 0] = (1.0 + dh / float(MAX_VALUE)) * hlsImg[:, :, 0]
    hlsImg[:, :, 0][hlsImg[:, :, 0] > 255] = 255
    hlsImg[:, :, 0][hlsImg[:, :, 0] < 0] = 0
    hlsImg = np.uint8(hlsImg)
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return lsImg


def saturation(img, ds):
    '''
    Adjust saturation.
    :param img:
    :param ds: (-100, 100)
    :return:
    '''
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hlsImg = np.uint16(hlsImg)
    MAX_VALUE = 100
    hlsImg[:, :, 2] = (1.0 + ds / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 255] = 255
    hlsImg[:, :, 2][hlsImg[:, :, 2] < 0] = 0
    hlsImg = np.uint8(hlsImg)
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return lsImg

def lightness(img, dl):
    '''
    Adjust lightness.
    :param img:
    :param dl: (-100, 100)
    :return:
    '''
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hlsImg = np.uint16(hlsImg)
    MAX_VALUE = 100
    hlsImg[:, :, 1] = (1.0 + dl / float(MAX_VALUE)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 255] = 255
    hlsImg[:, :, 1][hlsImg[:, :, 1] < 0] = 0
    hlsImg = np.uint8(hlsImg)
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return lsImg