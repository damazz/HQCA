import numpy as np
from hqca.core import *
from hqca.storage import *
from typing import Optional
from hqca.core.primitives import *
from hqca.operators import *
from qiskit.transpiler import Layout
from qiskit import transpile,assemble,QuantumRegister,QuantumCircuit,ClassicalRegister
from qiskit import Aer,execute
np.set_printoptions(precision=4,suppress=True)

class IterativeUnitarySimulator:
    def __init__(self,
            ):
        ''' Simulator to run an iterative unitary.

        Needs a quantum 
        '''
        pass


    def set(self,
            quantstore: QuantumStorage,
            initial=False,
            instruct=None,
            ):
        ''' Sets the simulator, constructs necessary circuits

        '''
        self.qs = quantstore
        i=0
        qc = GenericCircuit(
                QuantStore=quantstore,
                _name='psi',
                )
        if initial:
            qc = self._set_initial_state(qc)
        qc.apply(Instruct=instruct)
        self.circuit = qc.qc
        self.qr = qc.q
        self.cr = qc.c


    def _set_initial_state(self,qc):
        init = Operator()
        init+= PauliString('I'*self.qs.Nq,1)
        for n,item in enumerate(self.qs.initial):
            tem = Operator()
            op1 = FermiString(1,
                    indices=[item],
                    ops='+',
                    N=self.qs.dim)
            op2 = FermiString( -1,
                    indices=[item],
                    ops='-',
                    N=self.qs.dim,
                    )
            tem+=op1
            tem+=op2
            try:
                new = tem.transform(self.qs.initial_transform)
                init*= new
            except AttributeError:
                new = tem.transform(self.qs.transform)
                init*= new
        try:
            U = self.initial_clifford
            apply_clifford_operation(qc,U)
        except AttributeError as e:
            pass
        except Exception as e:
            print('Error in applying initial clifford transformation.')
            sys.exit(e)
        for s in init:
            apply_pauli_string(qc,s)
        return qc

    def simulate(self,
            tomo: Tomography,
            rho_i: Optional[np.ndarray]=None,
            ):
        self.circuit_list = []
        for op in tomo.op:
            self.circuit_list.append(op)
        beo = self.qs.beo
        if not self.qs.backend=='unitary_simulator':
            raise BackendError()
        #
        self.circuit = transpile(
                circuits=self.circuit,
                backend=beo,
                optimization_level=0,
                )
        #print(self.circuit)
        job = beo.run(self.circuit)

        U = job.result().get_unitary()

        #print('Unitary: ')
        self.rho = np.dot(U,np.dot(rho_i,np.conj(U.T)))
        counts = []
        #print('Running tomography')
        unitary_circs = []
        for ni,c in enumerate(self.circuit_list):
            #print(ni,c)
            qr = QuantumRegister(self.qs.Nq_tot)
            cr = ClassicalRegister(self.qs.Nq_tot)
            qc = QuantumCircuit(qr,cr)
            for n, i in enumerate(c):
                if i == 'X':
                    qc.h(qr[n])
                elif i == 'Y':
                    qc.sdg(qr[n])
                    qc.h(qr[n])
            unitary_circs.append(qc)
        us = Aer.get_backend('unitary_simulator')
        #print(unitary_circs)
        job = execute(unitary_circs,us,shots=1).result().results
        for n,c in enumerate(self.circuit_list[:]):
            U = job[n].data.unitary
            #print(U)
            ndiag = np.sqrt(np.diag(
                    np.dot(U,np.dot(self.rho,np.conj(U.T)))
                    ))
            counts.append(ndiag)
        #tomo.counts = counts
        #
        #
        tomo.counts = {i:j for i,j in zip(self.circuit_list,counts)}
        tomo.rho = self.rho
        #print(tomo.counts)


