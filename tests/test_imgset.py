from TransDict.imgset import CIFAR10_train, CIFAR10_test, Imagenet_val,CustomImgset,SingleImage
from TransDict.utils import clean_temp_dir
import unittest

class TestImgset(unittest.TestCase):
    def test_init_CIFAR10(self):
        cifar10_train = CIFAR10_train()
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
        custom.load_with_labels(src_dir='../data/custom', csv_file='../data/custom.txt', label_names=['aaa', 'bbb'])
        custom.add('random_resize', 256, 481)
        custom.add('random_crop', 224, 224)
        custom.output_todo_list()
        custom.run()
        custom.save('png','../data/output')
        custom.output_MT_history()

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