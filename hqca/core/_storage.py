from abc import ABC, abstractmethod

class Storage(ABC):
    '''
    Storage object. Contains 3 main methods.

    Evaluate: 
        Takes a new object and evaluates it versus the current Hamiltonian
        or relevant object. 

    Update:
        Updates the current, relevant parameters, such as updating the 2-RDM or
        other attributes. Basically, tracks the run. 

    Analysis:
        Method specific analysis. Prints out relevant information. 
    '''
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


