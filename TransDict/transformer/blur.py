import cv2

def mean_blur(img, ksize):
	# (a,b)表示的卷积核[大小]  a代表横向上的模糊，b表示纵向上的模糊
	return cv2.blur(img, ksize)

def median_blur(img, ksize):
	# 第二个参数是孔径的尺寸，一个大于1的奇数。
    return cv2.medianBlur(img, ksize)


def Gaussian_blur(img, ksize):
    return cv2.GaussianBlur(img,ksize,0)

