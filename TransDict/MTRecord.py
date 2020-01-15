class MT_Record(object):
    def __init__(self, MT_name, *args):
        self.MT_name = MT_name
        self.args = args

    def get_MT_name(self):
        return self.MT_name

    def get_args(self):
        return self.args

    def __str__(self):
        return self.MT_name + str(self.args)