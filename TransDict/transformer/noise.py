import random
import numpy as np

def sp_noise(img, prob):
  '''
  添加椒盐噪声
  prob:噪声比例
  '''
  output = img.copy()
  prob /= 2
  thres = 1 - prob
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
  return output

def Gaussian_noise(img, means, sigma, prob):
  '''
    添加高斯噪声
    mean : 均值
    var : 方差
  '''
  NoiseImg = np.uint16(img)
  rows = NoiseImg.shape[0]
  cols = NoiseImg.shape[1]
  for i in range(rows):
      for j in range(cols):
          rdn = random.random()
          if rdn < prob:
            NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma)

  NoiseImg = np.uint8(np.clip(NoiseImg, 0, 255))
  return NoiseImg