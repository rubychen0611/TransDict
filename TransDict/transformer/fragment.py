import numpy as np
def fragment(img, offset=7):
    '''
    Fragment effect.
    :param img: the original image
    :param offset: size of offsets
    :return: the transformed image
    '''
    img = np.uint16(img)
    img_1 = img.copy()
    img_2 = img.copy()
    img_3 = img.copy()
    img_4 = img.copy()
    row, col, channel = img.shape

    img_1[:, 0: col - 1 - offset, :] = img[:, offset:col - 1, :]
    img_2[:, offset:col - 1, :] = img[:, 0: col - 1 - offset, :]
    img_3[0:row - 1 - offset, :, :] = img[offset:row - 1, :, :]
    img_4[offset:row - 1, :, :] = img[0:row - 1 - offset, :, :]

    img_out = (img_1 + img_2 + img_3 + img_4) / 4.0
    return np.uint8(img_out)
