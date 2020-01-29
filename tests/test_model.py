import sys

import keras
import numpy as np
from keras.datasets import cifar10

sys.path.append(r'/home/czq/TransDict/')
from TransDict.imgset import CIFAR10_train, CIFAR10_test
from TransDict.model import Model
import unittest
class TestModel(unittest.TestCase):
    def test_cifar10_training(self):
        cifar10_train = CIFAR10_train()
        cifar10_test = CIFAR10_test()
        model = Model()
        model.train(cifar10_train, cifar10_test, 'ResNet')
        model.save('./ResNet20.h5')

        print("predict results: ")
        scores = model.predict(cifar10_test, cifar10_train.mean)
        print(scores)

    def test_cifar10_load(self):
        cifar10_train = CIFAR10_train()
        cifar10_test = CIFAR10_test()
        model = Model()
        model.load('../saved_models/ResNet20.h5')
        print("predict results: ")
        scores = model.predict(cifar10_test, cifar10_train.mean)
        print(scores)

    def test_cifar10_load1(self):
        _, (x_test, y_test) = cifar10.load_data()
        y_test = keras.utils.to_categorical(y_test, 10)
        m = Model()
        m.load('../model/cifar10/models/resnet.h5')
        temp = np.copy(x_test)
        temp = temp.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
        scores = m.model.evaluate(temp, y_test, verbose=1)
        print(scores)

    def test_cifar10_compare(self):
        cifar10_train = CIFAR10_train()
        cifar10_test0 = CIFAR10_test()
        cifar10_test1 = CIFAR10_test()
        model = Model()
        model.load('saved_models/ResNet20.h5')
        print("predict results: ")
        model.predict_comp(cifar10_test0, cifar10_test1, cifar10_train.mean)
        cifar10_test1.add('resize_size', 24, 24)
        cifar10_test1.add('resize_size', 32, 32)
        cifar10_test1.run()
        model.predict_comp(cifar10_test0, cifar10_test1, cifar10_train.mean)

if __name__ == "__main__":
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestModel('test_cifar10_compare'))

    # suite =  unittest.TestLoader().loadTestsFromTestCase(MyTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
