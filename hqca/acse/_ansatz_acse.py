from hqca.quantum.BuildCircuit import GenerateDirectCircuit
from hqca.tools import Functions as fx
from hqca.tools.RDMFunctions import check_2rdm
import numpy as np
from hqca.quantum.BuildCircuit import GenerateGenericCircuit
from hqca.quantum.primitives import _Tomo as tomo
from qiskit import Aer,IBMQ,execute
from qiskit.compiler import transpile
from hqca.quantum.Tomography import Tomography
from qiskit.compiler import assemble
from qiskit.tools.monitor import backend_overview,job_monitor
from hqca.tools.RDM import Recursive,RDMs
from hqca.quantum.primitives._Hamiltonian import _add_Hamiltonian
from hqca.quantum.primitives._Hamiltonian import _generic_Pauli_term
from copy import deepcopy as copy
import sys

class ACSEAnsatz(Operator):
    '''
    generate an ansatz for the ACSE ansatz

    where should this go?
    '''
    def __init__(self,
            trialAnsatz=False,
            propagateTime=,
            scalingHam=1.0,
            **kw
            ):
        #Tomography.__init__(self,QuantStore,**kw)
        self._ansatz = []
        self.scaleH = scalingHam
        self.propagate=propagateTime
        if propagateTime:
            self.qubOp = Store.qubOp[:]
        self.qubitH = Store.qubitH

    @property
    def ansatz(self):
        return self._op

    @ansatz.setter
    def ansatz(self,a):
        self._op = a


