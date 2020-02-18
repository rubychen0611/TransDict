from TransDict.imgset import CIFAR10_train, CIFAR10_test, Imagenet_val,CustomImgset,SingleImage
from TransDict.utils import clean_temp_dir
import unittest

class TestImgset(unittest.TestCase):
    def test_init_CIFAR10(self):
        cifar10_test = CIFAR10_test()
        cifar10_test.add('resize_size', 24, 24)
        cifar10_test.run()
        #print(cifar10_train.mean, cifar10_train.std)
        #cifar10_train.display()

    def test_imagenet_val(self):
        clean_temp_dir()
        imagenet_val = Imagenet_val(0,100)
        #imagenet_val.display()
        imagenet_val.save('jpg', './crop')

    def test_custom_Imgset(self):
        clean_temp_dir()
        custom = CustomImgset()
        custom.load_with_labels(src_dir='../data/custom',csv_file='../data/custom.txt', label_names=['aaa','bbb'])
        #custom.load_without_labels(src_dir='../data/custom')
        custom.display()

    def test_single_image(self):
        img = SingleImage()
        #img.load_without_label('../data/custom/1.jpg')
        img.add('crop', 100,200)
        img.add('resize', 200,300)
        img.run()

    def test_crop(self):
        clean_temp_dir()
        custom = CustomImgset()
        #custom.load_with_labels(src_dir='../data/custom', csv_file='../data/custom.txt', label_names=['aaa', 'bbb'])
        custom.load_without_labels('../data/custom')
        #custom.display()
        custom.add('random_resize', 256, 481)
        custom.add('random_crop', 224, 224)
        custom.output_todo_list()
        custom.run()
        custom.display()
        #custom.save('png','../data/output')
        custom.output_MT_history()

    def test_rotate(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        #custom.add('rotate', -1)
        custom.add('rotate_clockwise_90')
        custom.add('rotate_anticlockwise_90')
        custom.run()
        custom.display()

    def test_flip(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        #custom.add('flip_horizontal')
        custom.add('flip_vertical')
        custom.run()
        custom.display()

    def test_translate(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        custom.add('translate', -10, 20)
        custom.run()
        custom.display()

    def test_scale(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        custom.add('scale', 0.99)
        custom.run()
        custom.display()

    def test_brightness_contrast(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        custom.add('contrast', 0.5)
        #custom.add('brightness', 70)
        custom.run()
        custom.display()

    def test_blur(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        #custom.add('mean_blur', (30, 30))
        #custom.add('median_blur', 31)
        custom.add('Gaussian_blur', (29, 29))
        custom.run()
        custom.display()

    def test_mosaic(self):
        custom = CustomImgset()
        custom.load_without_labels('../data/custom')
        custom.add('mosaic', 40)
        custom.run()
        custom.display()

    def test_noise(self):
        img = SingleImage()
        img.load_without_label('../data/custom/9.jpg')
        img.add('Gaussian_noise', 0, 3, 0.99)
        img.run()
        img.display()

    def test_sharpen(self):
        img = SingleImage()
        img.load_without_label('../data/custom/11.jpg')
        img.add('USM_sharpen')
        img.run()
        img.display()
        #img.save('jpg', '../data/output')

    def test_fragment(self):
        img = SingleImage()
        img.load_without_label('../data/custom/11.jpg')
        img.add('fragment', 5)
        img.run()
        img.display()

    def test_saturation(self):
        img = SingleImage()
        img.load_without_label('../data/custom/9.jpg')
        img.add('saturation', -80)
        img.run()
        img.display()

    def test_lightness(self):
        img = SingleImage()
        img.load_without_label('../data/custom/9.jpg')
        img.add('lightness', -80)
        img.run()
        img.display()

    def test_hue(self):
        img = SingleImage()
        img.load_without_label('../data/custom/12.jpg')
        img.add('hue', 100)
        img.run()
        img.display()

    def test_temperature(self):
        img = SingleImage()
        img.load_without_label('../data/custom/10.jpg')
        img.add('color_temperature', 200)
        img.run()
        img.display()

    def test_temperature(self):
        img = SingleImage()
        img.load_without_label('../data/custom/12.jpg')
        img.add('exposure', 0.2)
        img.run()
        img.display()

if __name__ == "__main__":
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestImgset('test_init_CIFAR10'))
    #suite.addTest(TestImgset('test_imagenet_val'))
    #suite.addTest(TestImgset('test_custom_Imgset'))
    #suite.addTest(TestImgset('test_single_image'))
    #suite.addTest(TestImgset('test_crop'))

    # suite =  unittest.TestLoader().loadTestsFromTestCase(MyTest)
    unittest.TextTestRunner(verbosity=2).run(suite)