from abc import ABC, abstractmethod



class Instructions(ABC):
    '''
    Should input an ansatz, and then output the proper circuit instructions

    For default, maybe no action? 
    '''
    def __init__(self,
            Ansatz,
            procedure='default',
            **kw):
        self._gates = []


    @abstractmethod
    def convert_ansatz(self):
        pass


    @property
    @abstractmethod
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,gates):
        self._gates = gates


