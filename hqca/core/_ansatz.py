from abc import abstractmethod,ABC






class Ansatz(ABC):
    '''
    Defined for a certain method, and can take in parameters of some sort and
    make a new or temporary ansatz.
    '''

    @abstractmethod
    def __init__(parameters,
            self):
        pass
