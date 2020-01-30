import cv2
def rotate(img, angle):
    height = img.shape[0]
    width = img.shape[1]
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, 1)  # mat rotate 1 center 2 angle 3 缩放系数
    img = cv2.warpAffine(img, matRotate, (width, height))
    return img

def rotate_clockwise_90(img):
    '''
    rotate the image 90 degrees clockwise
    :param img: the orginal image
    :return: the rotated image
    '''
    img = cv2.transpose(img)
    img = cv2.flip(img, 1)
    return img

def rotate_anticlockwise_90(img):
    '''
    rotate the image 90 degrees anticlockwise
    :param img: the orginal image
    :return: the rotated image
    '''
    img = cv2.transpose(img)
    img = cv2.flip(img, 0)
    return img
