from qiskit import execute
from abc import ABC,abstractmethod
from qiskit.circuit import Parameter
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit




class GenerateCircuit(ABC):
    @abstractmethod
    def __init__(
            self,
            QuantStore,
            Instructions,
            _name=False,
            ):
        self.qs = QuantStore
        self.Nq = QuantStore.Nq_tot
        self.q = QuantumRegister(self.Nq,name='q')
        self.c = ClassicalRegister(self.Nq,name='c')
        self.Ne = QuantStore.Ne
        self.name = _name
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)
        self._initialize()
        for var,fxn in Instructions.gates:
            fxn(Q=self,*var)
