from abc import ABC, abstractmethod 

class QuantumRun(ABC):
    @abstractmethod
    def __init__(self,**kw):
        pass

    @abstractmethod
    def update(self,**kw):
        pass

    @abstractmethod
    def build(self,**kw):
        '''
        Make sure the run is compiled okay. 
        '''
        pass

    @abstractmethod
    def run(self,**kw):
        pass


