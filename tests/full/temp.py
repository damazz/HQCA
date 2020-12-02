import qiskit
import numpy as np
import sys
from math import pi
import qiskit.providers.aer.noise as noise
from deconstruct import *
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.processes import *
from hqca.acse import *
from hqca.core import *
from hqca.core.primitives import *
from pyscf import gto
from hqca.transforms import *
from functools import partial
from hqca.tools import *
from hqca.state_tomography import *
from tests._generic import *
from delayed_assert import delayed_assert as da
np.set_printoptions(precision=3)

class Ins(Instructions):
    def __init__(self,operator,
            Nq,propagate=False,**kw):
        try:
            operator = operator.op_form()
        except AttributeError as Exception:
            pass
        if Nq==2:
            self.use = []
            self._gates = []
            a1 = ['XY','YX']
            for op in operator:
                if op.s==a1[0]:
                    self.use.append('a1')
                if len(self.use)==3:
                    break
            get_funcs = {
                    'a1':self._type_A1,
                    }
            for u in self.use[:3]:
                self._gates.append(
                        [(operator,),get_funcs[u]]
                        )
        if propagate:
            self._applyH(Nq=Nq,**kw)

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a


    def _type_A1(self,Q,op):
        '''
        connectvitiy?

        0-1-2-3 (old)
        ----> 

        1-2-3-0
        '''
        paulis = ['YX','XY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(0)
        Q.s(1)
        Q.h(1)
        Q.Cx(1,0)
        Q.Ry(1,para[0])
        Q.Ry(0,-para[1])
        Q.Cx(1,0)
        Q.h(1)
        Q.si(1)
        Q.si(0)


    def _applyH(self,
            HamiltonianOperator,
            scaleH=0.5,Nq=2,**kw):
        self.hss = scaleH
        if Nq==2:
            self.use_H = ['h0']
            get_funcs = {
                    'h0':self._type_H0,
                    }
            for u in self.use_H[:]:
                self._gates.append(
                        [(HamiltonianOperator,),get_funcs[u]]
                        )

    def _type_H0(self,Q,op):
        h0 = ['ZI','IZ','ZZ','XX','YY']
        para = []
        for p in h0:
            for o in op:
                if o.s==p:
                    para.append(o.c.real*self.hss)
        #
        Q.Rz(0,para[0])
        Q.Rz(1,para[1])

        Q.Cx(0,1)
        Q.Rz(1,para[2])
        Q.Cx(0,1)

        Q.z(0)
        Q.s(1)
        Q.h(1)
        Q.Cx(1,0)
        Q.Ry(1,para[3])
        Q.Ry(0,para[4])
        Q.Cx(1,0)
        Q.h(1)
        Q.si(1)
        Q.z(0)


def test_noisy():
    ham = generic_molecular_hamiltonian()
    mol = generic_mol()
    s = Stabilizer(ham._qubOp,verbose=True)
    s.gaussian_elimination()
    s.find_symmetry_generators()
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['ZIIZ','ZIZ'],
            qubits=[3,2],
            eigvals=[-1,+1],
            )
    ham = MolecularHamiltonian(mol,transform=Tr)

    #
    #fermi = Operator([
    #    FermiString(coeff=0.5,ops='+-p',indices=[2,3,1],N=4),
    #    FermiString(coeff=0.5,ops='+-p',indices=[3,2,1],N=4),
    #    ])
    #print(fermi.transform(Tr))
    #sys.exit()

    st = StorageACSE(ham)
    qs = QuantumStorage()
    pr = StabilizerProcess('filter_diagonal')
    Nq = 2
    Ns = 8192
    error = 1.0
    qs.set_algorithm(st)
    qs.set_backend(
            backend='qasm_simulator',
            Nq=Nq,
            num_shots=Ns,
            provider='Aer')
    nm = model_v2(scaling=error)
    qs.set_noise_model(custom=True,
            noise_model=nm)
    qs.initial_transform=iTr
    proc = StabilizerProcess('filter_diagonal')
    qs.set_error_mitigation('ansatz_shift',coeff=1.0,D0=False)
    # run 1
    ins = PauliSet
    tomoRe = ReducedTomography(qs,verbose=False)
    tomoRe.generate(real=True,imag=True,
            simplify=True,transform=Tr,
            method='gt',strategy='lf')
    tomoIm = ReducedTomography(qs,verbose=False)
    tomoIm.generate(real=True,imag=True,
            simplify=True,transform=Tr,
            method='gt',strategy='lf')
    #st = StorageACSE(ham,closed_ansatz=-2)
    st = StorageACSE(ham,closed_ansatz=True)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='newton',
            update='quantum',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='norm',
            hamiltonian_step_size=0.5,
            max_iter=5,
            initial_trust_region=2.0,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tomoIm,
            tomo_Psi = tomoRe,
            verbose=True,
            )
    acse.build()
    da.expect(abs(acse.e0+0.783792654277353)<=1e-10)
    acse.run()
    #da.expect(abs(acse.e0+0.846147736093)<=1e-8)
    da.assert_expectations()
    #   #
    #  #
    # # 
    ##
test_noisy()
