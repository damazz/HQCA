from abc import ABC, abstractmethod

class Process(ABC):
    '''
    Process class.

    Takes input and will process accordingly.

    Useful for dealing wiht stabilizer codes, ancilla, and other forms of error
    mitigation.
    '''
    def __init__(self,**kw):
        self._data = None

    @abstractmethod
    def process(self,**kw):
        pass
