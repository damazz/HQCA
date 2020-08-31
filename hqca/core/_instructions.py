from abc import ABC, abstractmethod



class Instructions(ABC):
    '''

    Instructions, or Instruct class.

    Takes an input operator, and then follows a standard set of instructions to
    process the operator and generate a gates object.

    For default, maybe no action? 
    '''
    def __init__(self,
            **kw):
        self._gates = []

    @property
    @abstractmethod
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,gates):
        self._gates = gates
