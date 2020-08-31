from qiskit import execute
from abc import ABC,abstractmethod
from qiskit.circuit import Parameter
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit


class Circuit(ABC):
    @abstractmethod
    def __init__(self,
            QuantStore,
            _name=False,
            ):
        self.qs = QuantStore
        self.Nq = QuantStore.Nq_tot
        self.q = QuantumRegister(self.Nq,name='q')
        self.c = ClassicalRegister(self.Nq,name='c')
        self.name = _name
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)

    @abstractmethod
    def apply(self,
            Instruct,
            ):
        for var,fxn in Instruct.gates:
            fxn(self,*var)

    @abstractmethod
    def tomography(self,
            Instruct,
            ):
        for var,fxn in Instruct.generate_tomography(self):
            fxn(self,*var)


