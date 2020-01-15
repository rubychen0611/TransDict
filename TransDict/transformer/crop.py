import random
from TransDict.core.exceptions import TransformationError

def crop(img, left, top, crop_width, crop_height):
    '''
    Crop the image according to the start point and the target size.
    :param img: original image
    :param left: start point's y
    :param top:  start point's x
    :param crop_width: target width
    :param crop_height: target height
    :return: cropped image
    '''
    height = img.shape[0]
    width = img.shape[1]
    if left + crop_width > width or top + crop_height > height:
        raise TransformationError('Cannot crop the image into the target size.')
    img = img[top:top + crop_height, left:left + crop_width]
    return img

def random_crop(img, crop_width,  crop_height):
    '''
    Randomly crop the image into the target size.
    :param img: original image
    :param crop_width: target width
    :param crop_height: target height
    :return: cropped image
    '''
    height = img.shape[0]
    width = img.shape[1]
    if height < crop_height or width < crop_width:
        raise TransformationError('Cannot crop the image into a bigger size.')
    if height > crop_height:
        top = random.randint(0, height - crop_height)
    else:
        top = 0
    if width > crop_width:
        left = random.randint(0, width - crop_width)
    else:
        left = 0
    img = img[top:top + crop_height, left:left + crop_width]
    return img