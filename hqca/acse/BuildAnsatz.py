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
import sys

class Ansatz(Tomography):
    '''
    generate a tomography for the ACSE ansatz
    '''
    def __init__(self,
            Store,
            QuantStore,
            trialAnsatz=False,
            propagateTime=False,
            scalingHam=1.0,
            Hamiltonian='standard',
            **kw
            ):
        Tomography.__init__(self,QuantStore,**kw)
        if trialAnsatz:
            self.S = Store.tempAnsatz
        else:
            self.S = Store.ansatz
        self.scaleH = scalingHam
        self.propagate=propagateTime
        if propagateTime:
            if Hamiltonian=='standard':
                self.qubOp = Store.qubOp[:]
            elif Hamiltonian=='split-K':
                pass
            elif Hamiltonian=='split-V':
                pass
        self.qubitH = Store.qubitH

    def build_tomography(self,**kw):
        Tomography.generate_2rdme(self,**kw)
        self._gen_full_tomography(**kw)

    def _applyH(self,Q,trotter_steps):
        if self.qubitH=='qiskit':
            iters = 1
            for i in range(iters):
                _add_Hamiltonian(Q,self.qubOp,scaling=self.scaleH/iters)
        elif self.qubitH=='local':
            for i in range(trotter_steps):
                for pauli,coeff in zip(self.qubOp[0],self.qubOp[1]):
                    _generic_Pauli_term(
                            Q,(1/trotter_steps)*coeff,pauli,scaling=self.scaleH)

    def _gen_full_tomography(self,trotter_steps=1,**kw):
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenerateGenericCircuit(
                    self.qs,
                    self.S,
                    _name=circ)
            if self.propagate:
                self._applyH(Q,trotter_steps)
            for n,q in enumerate(circ):
                tomo._apply_pauli_op(Q,n,q)
            if self.qs.backend in ['unitary_simulator','statevector_simulator']:
                pass
            else:
                Q.qc.measure(Q.q,Q.c)
            self.circuits.append(Q.qc)

    def _gen_quantum_S(self):
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenerateDirectCircuit(self.qs,_name=circ)
            if self.propagate:
                self._applyH(Q)
            for n,q in enumerate(circ):
                tomo._apply_pauli_op(Q,n,q)
            if self.qs.backend in ['unitary_simulator','statevector_simulator']:
                pass
            else:
                Q.qc.measure(Q.q,Q.c)
            self.circuits.append(Q.qc)
