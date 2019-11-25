from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def __init__(self,
            hamiltonian=None):
        pass

    @abstractmethod
    def evaluate(self,**kw):
        pass
    
    @abstractmethod
    def update(self,**kw):
        pass

    @abstractmethod
    def analysis(self,**kw):
        pass




class QuantumStorage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_backend(self):
        pass

