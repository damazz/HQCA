import warnings

class QuantumRunError(Exception):
    '''
    Generic error for quantum runs. 
    '''
    pass

class DeviceConfigurationError(Exception):
    '''
    Raised for incompatibilities in configurations. 
    '''
    pass

class KeywordError(Exception):
    pass

class TransformError(Exception):
    pass

class TomographyError(Exception):
    pass

class PositivityError(Exception):
    pass

