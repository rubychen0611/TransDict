import datetime
import random
import shutil
import sys
import demjson
import keras
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
from keras.datasets import cifar10, cifar100

from TransDict import utils
from TransDict.MTRecord import MT_Record
from TransDict.transformer import *
from .core import ClassInfoError, NoLabelsError, UnknownFormatError, ImageLoadingError, EmptySetError, Logger

MAX_RAM_LIMIT = 4 * 1024  # MB
switcher = {"crop": crop, "random_crop":random_crop, "resize_size": resize_size, "resize_scale": resize_scale,
            "random_resize": random_resize, "rotate": rotate, "rotate_clockwise_90": rotate_clockwise_90,
            "rotate_anticlockwise_90": rotate_anticlockwise_90, "flip_horizontal":flip_horizontal,
            "flip_vertical": flip_vertical, "translate": translate, "scale": scale,"brightness":brightness,
            "contrast":contrast, "mean_blur":mean_blur, "median_blur":median_blur, "Gaussian_blur":Gaussian_blur,
            "mosaic":mosaic, "sp_noise":sp_noise, "Gaussian_noise":Gaussian_noise, "USM_sharpen":USM_sharpen,
            "fragment":fragment, "saturation":saturation, "lightness": lightness, "color_temperature":color_temperature,
            "hue":hue}
class ClassInfo(object):
    '''Basic information of classes'''

    def __init__(self, class_names):
        if not isinstance(class_names, list):
            raise ClassInfoError('Wrong type of class_names.')
        self.class_names = class_names  # names of each classes
        self.n_classes = len(class_names)  # number of classes

    def get_n_classes(self):
        return self.n_classes

    def get_class_names(self):
        return self.class_names


class Imgset(object):
    """Base class representing a image dataset."""
    def __init__(self):
        self.img_names = []  # file name
        self.images = []  # images (when stored in memory)
        self.img_dir = ""  # image dir (when stored in disk)
        self.with_labels = False
        self.in_memory = True
        self.labels = []  # (when has labels)
        self.class_info = []  # (when has labels)
        self.logger = Logger.getInstance()
        self.todo_MT_list = []
        self.MT_history = []

    def display(self, start_idx=0):
        
        if self.get_size() == 0:
            raise EmptySetError('The image set is empty.')
        MAX_EDGE_SIZE = 250

        def resize_img(img):
            w = img.width
            h = img.height
            if w < h:
                w = (int)(w * MAX_EDGE_SIZE / h)
                h = MAX_EDGE_SIZE
            else:
                h = (int)(h * MAX_EDGE_SIZE / w)
                w = MAX_EDGE_SIZE
            img = img.resize((w, h))
            return img

        window = tk.Tk()
        window.title('Image Viewer')
        window.geometry('400x400')
        img = self.get_img(start_idx)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        tkImage = ImageTk.PhotoImage(resize_img(img))
        img_view = tk.Label(window, image=tkImage)
        img_view.pack(expand='yes')

        frame1 = tk.Frame(window)
        img_idx_var = tk.StringVar()
        img_idx_var.set(value=str(start_idx))
        img_idx = tk.Label(frame1, textvariable=img_idx_var)
        img_idx.pack()
        img_name_var = tk.StringVar()
        img_name_var.set(value="File name: " + self.get_img_name(start_idx))
        img_name = tk.Label(frame1, textvariable=img_name_var)
        img_name.pack()
        if self.with_labels:
            img_label_var = tk.StringVar()
            img_label_var.set(value="Label: " + self.get_label(start_idx))
            img_label = tk.Label(frame1, textvariable=img_label_var)
            img_label.pack()

        def on_click_left():
            cur_idx = (int)(img_idx_var.get())
            cur_idx = (cur_idx - 1) % self.get_size()
            img_idx_var.set(str(cur_idx))
            if self.with_labels:
                img_label_var.set(value="Label: " + self.get_label(cur_idx))
            img_name_var.set("File name: " + self.get_img_name(cur_idx))
            img = self.get_img(cur_idx)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            tkImage = ImageTk.PhotoImage(resize_img(img))
            img_view.configure(image=tkImage)
            img_view.image = tkImage
            window.update()

        def on_click_right():
            cur_idx = (int)(img_idx_var.get())
            cur_idx = (cur_idx + 1) % self.get_size()
            img_idx_var.set(str(cur_idx))
            if self.with_labels:
                img_label_var.set(value="Label: " + self.get_label(cur_idx))
            img_name_var.set("File name: " + self.get_img_name(cur_idx))
            img = self.get_img(cur_idx)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            tkImage = ImageTk.PhotoImage(resize_img(img))
            img_view.configure(image=tkImage)
            img_view.image = tkImage
            window.update()

        frame2 = tk.Frame(window)
        left_button = tk.Button(frame2, text='<', width=2, height=1, command=on_click_left)
        left_button.grid(row=0, column=1)
        right_button = tk.Button(frame2, text='>', width=2, height=1, command=on_click_right)
        right_button.grid(row=0, column=2)
        frame2.pack(side='bottom', pady=10)
        frame1.pack(side='bottom')
        window.mainloop()

    def save(self, dst_format, dst_dir):
        '''
        Save the image set to dst_dir with dst_format
        :param dst_format: 'png' or 'jpg'
        :param dst_dir: a destination directory
        '''
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        if dst_format == 'jpg':
            self.logger.info('Start saving images to %s.' % dst_dir)
            for i in range(self.get_size()):
                img = self.get_img(i)
                cv2.imwrite(dst_dir + '/' + self.get_img_name(i) + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            self.logger.info('Finish saving images.')
        elif dst_format == 'png':
            self.logger.info('Start saving images to %s.' % dst_dir)
            for i in range(self.get_size()):
                img = self.get_img(i)
                cv2.imwrite(dst_dir + '/' + self.get_img_name(i) + '.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            self.logger.info('Finish saving images.')
        else:
            raise UnknownFormatError('Unknown format \'%s\', only \'png\' and \'jpg\' are supported.' % dst_format)

    def get_img(self, idx):
        if self.in_memory == True:
            return self.images[idx]
        else:
            imgpath = self.img_dir + '/' + self.get_img_name(idx) + '.npy'
            img = np.load(imgpath)
            return img

    def get_img_name(self, idx):
        return self.img_names[idx]

    def get_label(self, idx):
        if self.with_labels == False:
            raise NoLabelsError('The image set does not have labels provided.')
        return self.class_info.get_class_names()[self.labels[idx]]

    def get_size(self):
        return len(self.img_names)

    def output_todo_list(self):
        '''Check the todo_list'''
        for record in self.todo_MT_list:
            print(record)

    def output_MT_history(self):
        '''Check the MT history'''
        for record in self.MT_history:
            print(record)


    def add(self, MT_name, *args):
        self.todo_MT_list.append(MT_Record(MT_name, *args))


    def run(self):
        if not self.in_memory:
            old_dir = self.img_dir
            new_dir = self.gen_temp_dir()
            self.logger.info('Create a new temp dir ' + new_dir)
        self.logger.info('Start running transformation ...')
        for idx in range(self.get_size()):
            img = self.get_img(idx)
            for record in self.todo_MT_list:
                img = switcher.get(record.get_MT_name())(img, *record.get_args())
            if not self.in_memory:
                os.remove(os.path.join(old_dir, self.get_img_name(idx)) + '.npy')
                np.save(os.path.join(new_dir, self.get_img_name(idx)), img)
            else:
                self.images[idx] = img
        self.logger.info('Successfully finished transformation.')
        if not self.in_memory:
            self.logger.info('Remove the old temp dir '+ old_dir)
            os.rmdir(old_dir)
            self.img_dir = new_dir

        # update todo_list and history
        self.MT_history += self.todo_MT_list
        self.todo_MT_list.clear()
        self.logger.debug('Updated the MT history.')

    # def random_crop(self, img, crop_width=224, crop_height=224):
    #     '''Crop the image into 224*224 according to the ResNet paper'''
    #     height = img.shape[0]
    #     width = img.shape[1]
    #     # resize
    #     random_s = random.randint(256, 481)
    #     if height < width:
    #         resized_height = random_s
    #         resized_width = int(width * resized_height / height)
    #     else:
    #         resized_width = random_s
    #         resized_height = int(height * resized_width / width)
    #     img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    #     # crop
    #     top = random.randint(0, resized_height - crop_height)
    #     left = random.randint(0, resized_width - crop_width)
    #     img = img[top:top + crop_height, left:left + crop_width]
    #     return img

    def gen_temp_dir(self):
        '''
        Generate an unique ID for a temp directroy
        :return: unique id
        '''
        #local_time = time.localtime(time.time())
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        path = utils.TEMP_DIR + time_stamp
        os.makedirs(path)
        self.logger.info('Create temp directory: ' + path)
        return path

    def clean_cur_temp_dir(self):
        '''Clean the current temp directory that stored images'''
        if self.img_dir != '' and os.path.exists(self.img_dir):
            shutil.rmtree(self.img_dir)
        self.img_dir = ''

    def preprocess(self, x, y, mean):
        x_copy = x[:]
        y_copy = y[:]
        #x_copy = x_copy.astype('float32')
        
        #x_copy -= mean
        # Normalize data.
        for i in range(len(x_copy)):
            x_copy[i] = x_copy[i].astype('float32')
            x_copy[i] -= mean
            x_copy[i] = x_copy[i]/255
        #x_copy = x_copy.astype('float32') / 255
        y_copy = keras.utils.to_categorical(y_copy, self.class_info.get_n_classes())
        return x_copy, y_copy

    def cal_mean(self):
        if self.in_memory:
            self.mean = np.mean(self.images, axis=0)
        else:
            pass



class CustomImgset(Imgset):
    def __init__(self):
        super().__init__()
    def load_with_labels(self, src_dir, csv_file, label_names):
        '''
        Load images from a csv file with labels
        :param src_dir: source directory that stored images
        :param csv_file: a csv file that has two column, the first is filenames and the second is the corresponding labels.
        :param label_names: a list of label names
        :return:
        '''
        if self.get_size() != 0:
            self.clean_cur_temp_dir()
            super().__init__()
        self.with_labels = True
        self.class_info = ClassInfo(label_names)
        files = np.loadtxt(csv_file, str, delimiter=",", usecols=(0,1))
        # Memory size estimation
        img1 = cv2.imread(os.path.join(src_dir, files[0][0]))
        space = len(files) * sys.getsizeof(img1)  # B
        if space / (1024 * 1024) > MAX_RAM_LIMIT:
            self.logger.info("Large dataset, saved in disk.")
            self.in_memory = False
            self.img_dir = self.gen_temp_dir()
            for data in files:
                img_name = data[0][0].split('.')[0]
                self.img_names.append(img_name)
                img = cv2.imread(os.path.join(src_dir, data[0]))
                np.save(os.path.join(self.img_dir, img_name), img)
                label = int(data[1])
                if not (0 <= label < self.class_info.get_n_classes()):
                    raise ImageLoadingError('Label %d is out of indexes' % label)
                self.labels.append(label)
            self.logger.info("Finish loading.")
        else:
            self.logger.info("Small dataset, saved in memory.")
            for data in files:
                self.img_names.append(data[0][0].split('.')[0])
                self.images.append(cv2.imread(os.path.join(src_dir, data[0])))
                self.labels.append(int(data[1]))
                label = int(data[1])
                if not (0 <= label < self.class_info.get_n_classes()):
                    raise ImageLoadingError('Label %d is out of indexes' % label)
                self.labels.append(label)
            self.logger.info("Finish loading.")


    def load_without_labels(self, src_dir='../data/custom'):
        '''
        Load images from a directory without labels.
        :param src_dir: source directory that stored images
        :return:
        '''
        if self.get_size() != 0:
            self.clean_cur_temp_dir()
            super().__init__()
        self.with_labels = False
        files = os.listdir(src_dir)
        if len(files) == 0:
            raise ImageLoadingError('Empty folder.')
        # Memory size estimation
        img1 = cv2.imread(os.path.join(src_dir, files[0]))
        space = len(files) * sys.getsizeof(img1)  # B
        if space / (1024 * 1024) > MAX_RAM_LIMIT:
            self.logger.info("Large dataset, saved in disk.")
            self.in_memory = False
            self.img_dir = self.gen_temp_dir()
            for file in os.listdir(src_dir):
                img_name = file.split('.')[0]
                self.img_names.append(img_name)
                img = cv2.imread(os.path.join(src_dir, file))
                np.save(os.path.join(self.img_dir, img_name), img)
            self.logger.info("Finish loading.")
        else:
            self.logger.info("Small dataset, saved in memory.")
            for file in os.listdir(src_dir):
                self.img_names.append(file.split('.')[0])
                self.images.append(cv2.imread(os.path.join(src_dir, file)))
            self.logger.info("Finish loading.")


class SingleImage(Imgset):
    def __init__(self):
        super().__init__()

    def load_without_label(self, filename):
        if self.get_size() != 0:
            super().__init__()
        self.with_labels = False
        self.img_names.append(os.path.basename(filename).split('.')[0])
        self.images.append(cv2.imread(filename))
        self.logger.info("Successfully loading the single image.")

    def load_with_label(self, filename, label, label_names):
        if self.get_size() != 0:
            super().__init__()
        self.with_labels = True
        self.img_names.append(os.path.basename(filename).split('.')[0])
        self.images.append(cv2.imread(filename))
        self.class_info = ClassInfo(label_names)
        if not (0 <= label < self.class_info.get_n_classes()):
            raise ImageLoadingError('Label %d is out of indexes' % label)
        self.labels.append(label)

class CIFAR10_train(Imgset):
    '''CIFRAR-10 training dataset'''

    def __init__(self, start=0, end=50000):
        super().__init__()
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.class_info = ClassInfo(label_names)
        self.with_labels = True
        self.in_memory = True
        self.img_names = []
        for idx in range(start, end):
            self.img_names.append("%05d" % (idx))

        (images_np, self.labels), _ = cifar10.load_data()
        images_np = images_np[start:end, :, :, :]
        images_np = images_np[..., ::-1]  # RBG to BGR
        self.images = []
        for img in images_np:
            self.images.append(img)
        self.labels = self.labels[start:end, :]
        self.labels = self.labels.reshape(len(self.labels))
        self.cal_mean()

class CIFAR10_test(Imgset):
    '''CIFRAR-10 testing dataset'''
    def __init__(self, start=0, end=10000):
        super().__init__()
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.class_info = ClassInfo(label_names)
        self.with_labels = True
        self.in_memory = True
        self.img_names = []
        for idx in range(start, end):
            self.img_names.append("%05d" % (idx))

        _, (images_np, self.labels) = cifar10.load_data()
        images_np = images_np[start:end, :, :, :]
        images_np = images_np[..., ::-1]  # RBG to BGR
        self.images = []
        for img in images_np:
            self.images.append(img)
        self.labels = self.labels[start:end, :]
        self.labels = self.labels.reshape(len(self.labels))


class Imagenet_val(Imgset):
    def __init__(self, start=0, end=50000):
        super().__init__()
        label_name_file = open('../data/Imagenet/imagenet1000_clsidx_to_labels.txt')
        label_names = list(demjson.decode(label_name_file.read()).values())
        self.with_labels = True
        self.in_memory = False
        self.class_info = ClassInfo(label_names)
        self.img_addrs = []
        self.labels = []
        src_dir = '../data/Imagenet/ILSVRC2012_img_val/'
        src_file = open('../data/Imagenet/val.txt')
        img_names = []
        labels = []
        line = src_file.readline()
        while line:
            line = line.replace('\n', '')
            line_contents = line.split(' ')
            img_names.append(line_contents[0])
            labels.append(int(line_contents[1]))
            line = src_file.readline()
        src_file.close()
        self.labels = labels[start: end]
        self.img_names = img_names[start:end]
        self.img_dir = self.gen_temp_dir()
        self.logger.info('Start loading and cropping images...')
        for i in range(start, end):
            img = cv2.imread(src_dir + self.img_names[i])
            self.img_names[i] = self.img_names[i].split('.')[0]
            img = random_resize(img, 256, 481)
            img = random_crop(img, 224, 224)
            np.save(self.img_dir + '/' + self.img_names[i], img)
        self.logger.info('Finish loading images.')
