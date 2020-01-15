import warnings
import os
import shutil
import numpy as np
TEMP_DIR = './temp/'
def to_categorical(y, nb_classes, num_classes=None):
  """
  Converts a class vector (integers) to binary class matrix.
  This is adapted from the Keras function with the same name.
  :param y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
  :param nb_classes: nb_classes: total number of classes.
  :param num_classses: depricated version of nb_classes
  :return: A binary matrix representation of the input.
  """
  if num_classes is not None:
    if nb_classes is not None:
      raise ValueError("Should not specify both nb_classes and its deprecated "
                       "alias, num_classes")
    warnings.warn("`num_classes` is deprecated. Switch to `nb_classes`."
                  " `num_classes` may be removed on or after 2019-04-23.")
    nb_classes = num_classes
    del num_classes
  y = np.array(y, dtype='int').ravel()
  n = y.shape[0]
  categorical = np.zeros((n, nb_classes))
  categorical[np.arange(n), y] = 1
  return categorical

def clean_temp_dir():
  if os.path.exists(TEMP_DIR):
      shutil.rmtree(TEMP_DIR)  # 能删除该文件夹和文件夹下所有文件
  os.mkdir(TEMP_DIR)