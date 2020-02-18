import numpy as np

def mosaic(img, nsize):
    height = img.shape[0]
    width = img.shape[1]
    # 划分小方块，每个小方块填充随机颜色
    for y in range(0,height,nsize):
        for x in range(0,width,nsize):
            img[y:y+nsize,x:x+nsize] = img[y,x]
    return img
