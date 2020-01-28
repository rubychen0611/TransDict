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
        self.model.evaluate(x_test, y_test, verbose=1)

    def predict(self, imgset0, imgset1):
        pass

    def get_params(self):
        pass


