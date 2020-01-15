import random

import cv2


def resize_size(img, resized_width, resized_height):
    '''
    Resize the image according to the target width and height.
    :param img: original image
    :param width: resized width
    :param height: resized height
    :return:
    '''
    img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    return img

def resize_scale(img, fx, fy):
    '''

    :param img:
    :param fx:
    :param fy:
    :return:
    '''
    img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return img

def random_resize(img, min_s, max_s):
    '''

    :param img:
    :param min_s:
    :param max_s:
    :return:
    '''
    random_s = random.randint(min_s, max_s)
    height = img.shape[0]
    width = img.shape[1]
    if height < width:
        resized_height = random_s
        resized_width = int(width * resized_height / height)
    else:
        resized_width = random_s
        resized_height = int(height * resized_width / width)
    img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    return img
