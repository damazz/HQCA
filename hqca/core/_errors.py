"""
hqca errors.
"""

class OptimizationError(Exception):
    pass

class QuantumRunError(Exception):
    """
    Generic error for quantum runs.
    """
    pass

class QuantumRunBuildError(Exception):
    '''
    Error in building QuantumRun object. 
    '''
    pass

class ResidualError(Exception):
    '''
    '''

class HamiltonianError(Exception):
    pass

class DeviceConfigurationError(Exception):
    """
    Raised for incompatibilities in configurations.
    """
    pass


class KeywordError(Exception):
    """
    Missing certain keyword.
    """
    pass


class TransformError(Exception):
    """
    Error in performing operator transformations.
    """
    pass


class OperatorError(Exception):
    """
    Error in handling of operators
    """
    pass


class TomographyError(Exception):
    """
    Error in generation of tomography object.
    """
    pass


class PositivityError(Exception):
    """
    Error in the RDM
    """
    pass

class AnsatzError(Exception):
    """

    """
    pass


