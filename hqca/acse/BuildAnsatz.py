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
import sys

class Ansatz(Tomography):
    '''
    genearte a tomography for the ACSE ansatz
    '''
    def __init__(self,
            Store,
            QuantStore,
            trialAnsatz=False,
            propagateTime=False,
            scalingHam=1.0
            ):
        Tomography.__init__(self,QuantStore)
        if trialAnsatz:
            self.S = Store.tempAnsatz
        else:
            self.S = Store.ansatz
        self.scaleH = scalingHam
        self.propagate=propagateTime
        if propagateTime:
            self.qubOp = Store.qubOp


    def build_tomography(self,**kw):
        self._adapt_S()
        Tomography.generate_2rdme(self,**kw)
        self._gen_full_tomography()

    def applyS(self,Q):
        pass

    def _applyH(self,Q):
        _add_Hamiltonian(Q,self.qubOp,scaling=self.scaleH)
        # now, need to add to measurements

    def __processH(self):
        for term in self.qubOp:
            pauliString = term[1].to_label()
            add = False
            for i in pauliString:
                if i in ['X','x','Y','y']:
                    add=True

    def _gen_full_tomography(self):
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenerateGenericCircuit(
                    self.qs,
                    self.S,
                    _name=circ)
            if self.propagate:
                self._applyH(Q)
            for n,q in enumerate(circ):
                tomo._apply_pauli_op(Q,n,q)
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
            Q.qc.measure(Q.q,Q.c)
            self.circuits.append(Q.qc)

    #def _gen_tomo_list(self):
    #    self.op = []
    #    self.op.append('z'*self.Nq)
    #    for s in self.S:
    #        Tomography._gen_pauli_str(self,s)

    def _adapt_S(self):
        '''
        put S into qubit language
        '''
        for item in self.S:
            item.generateExcitationOperators(Nq=self.qs.Nq_tot)



