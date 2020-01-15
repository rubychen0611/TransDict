from TransDict.imgset import CIFAR10_train, CIFAR10_test
from TransDict.model import Model
import unittest
class TestModel(unittest.TestCase):
    def test_cifar10_training(self):
        cifar10_train = CIFAR10_train()
        cifar10_test = CIFAR10_test()
        model = Model()
        model.train(cifar10_train, cifar10_test, 'ResNet')


if __name__ == "__main__":
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestModel('test_cifar10_training'))

    # suite =  unittest.TestLoader().loadTestsFromTestCase(MyTest)
    unittest.TextTestRunner(verbosity=2).run(suite)