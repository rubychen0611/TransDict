import cv2
def flip_horizontal(img):
    '''
    flip the image horizontally
    :param img: the orginal image
    :return: the flipped image
    '''
    img = cv2.flip(img, 1)
    return img

def flip_vertical(img):
    '''
    flip the image vertically
    :param img: the orginal image
    :return: the flipped image
    '''
    img = cv2.flip(img, 0)
    return img