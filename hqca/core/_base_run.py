from abc import ABC, abstractmethod 

class QuantumRun(ABC):
    @abstractmethod
    def __init__(self,**kw):
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


class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False
