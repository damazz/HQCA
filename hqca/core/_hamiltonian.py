from abc import ABC, abstractmethod

class Hamiltonian(ABC):
    '''
    Hamiltonian object - in particular has two main properties:
        1) matrix (mandatory)
        2) operators (optional)

    The operators can be fermionic or qubit, and a method converting fermionic
    operators to qubit operators based on differeing transformations exists. 
    '''
    def __init__(self,
            matrix=None,
            fermiOperator=None,
            qubitOperator=None,
            model=None,
            **kw):
        self._matrix = matrix
        self._ferOp = fermiOperator
        self._qubOp = qubitOperator
        self._model = model
        self._order = 2


    @property
    @abstractmethod
    def matrix(self):
        return self._matrix

    @matrix.setter
    @abstractmethod
    def matrix(self,mat):
        self._matrix = mat

    @property
    @abstractmethod
    def model(self):
        return self._model

    @model.setter
    @abstractmethod
    def model(self,mod):
        self._model = mod

    @property
    def fermi_operator(self,**kw):
        return self._ferOp
    
    @fermi_operator.setter
    def fermi_operator(self,operator):
        self._ferOp = operator
    
    @property
    def qubit_operator(self):
        return self._qubOp
    
    @qubit_operator.setter
    def qubit_operator(self,operator):
        self._qubOp = operator

    def fermionic_to_qubit_operator(self,
            transformation='jordan-wigner'
            ):
        '''
        populates the qubit operators 
        '''
        newOp = []
        for op in self._ferOp:
            ferOp.generateQubitOperators(transformation)
            newOp.append(ferOp.ops)
        self._qubOp = newOp
        
