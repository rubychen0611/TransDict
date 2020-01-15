
class ClassInfoError(Exception):
    r"""Thrown when
    """
    pass
class ImageLoadingError(Exception):
    r"""Thrown when a image could not be loaded.
    """
    pass

class NoLabelsError(Exception):
    '''Thrown when performing label-related operations to an imageset without labels'''
    pass

class UnknownFormatError(Exception):
    '''Thrown when the input format is unknown'''
    pass

class EmptySetError(Exception):
    '''Thrown when an image set is empty'''
    pass

class TransformationError(Exception):
    '''Thrown when the transformation fails'''
    pass

class ModelTrainingError(Exception):
    pass