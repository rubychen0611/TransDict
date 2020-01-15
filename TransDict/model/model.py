from TransDict.model.resnet import ResNet
class Model(object):
    def __init__(self):
        self.model = None
        self.image_size = None

    def load(self, finename):
        pass

    def save(self, filename):
        pass

    def train(self, train_set, val_set, NN_type, *args):
        if NN_type == 'ResNet':
            self.model = ResNet(train_set, val_set)

    def predict(self, test_imgset):
        pass

    def predict(self, imgset0, imgset1):
        pass

    def get_params(self):
        pass
