from keras.models import load_model

from TransDict.model.resnet import ResNet
class Model(object):
    def __init__(self):
        self.model = None
        self.image_size = None

    def load(self, filename):
        self.model = load_model(filename)

    def save(self, filename):
        self.model.save(filename)

    def train(self, train_set, val_set, NN_type, *args):
        if NN_type == 'ResNet':
            self.model = ResNet(train_set, val_set)

    def predict(self, testset, mean):
        x_test, y_test = testset.preprocess(testset.images, testset.labels, mean)
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        return scores

    def predict_comp(self, imgset0, imgset1, mean):
        '''
        Compare the prediction results before and after metamorphic transformation.
        :param imgset0: the original image set
        :param imgset1: the metamorphic image set
        :param mean: the mean of training set, for preprocessing.
        :return: comparision results
        '''
        x_0, y_0 = imgset0.preprocess(imgset0.images, imgset0.labels, mean)
        scores0 = self.model.evaluate(x_0, y_0, verbose=1)
        preds0 = self.model.predict(x_0)
        print("before:")
        print(scores0)

        x_1, y_1 = imgset1.preprocess(imgset1.images, imgset1.labels, mean)
        scores1 = self.model.evaluate(x_1, y_1, verbose=1)
        preds1 = self.model.predict(x_1)
        print("after:")
        print(scores1)

        # difference between preds0 and preds1
        count = 0
        for i in range(preds0):
            if preds0[i] != preds1[i]:
                count += 1
        print("difference: ")

    def get_params(self):
        pass


