import numpy as np
from copy import deepcopy as copy
from collections import Counter
import sys
from functools import reduce
from hqca.core import *
from hqca.tools import *
from hqca.cqe._pauli_ansatz import *

class StorageHCSE(Storage):
    '''
    Storage object for use in ACSE calculation. In general, needs a Hamiltonian

    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,
            Hamiltonian=None,
            use_initial=False,
            closed_ansatz=True,
            ansatz=None,
            rdm = None,
            **kwargs):
        self.H = Hamiltonian
        self.rdm_coeff = 1
        self.p = Hamiltonian.order
        try:
            self.r = self.H.No_tot*2 # spin orbitals
        except:
            pass
        if type(ansatz)==type(None):
            ansatz = PauliAnsatz
        self.ansatz = []
        self.use_initial=use_initial
        if self.H.model in ['mol','molecular','molecule']:
            self.No_as = self.H.No_as
            self.Ne_as = self.H.Ne_as
            self.alpha_mo = self.H.alpha_mo
            self.beta_mo  = self.H.beta_mo
            if not use_initial:
                self._get_HF_rdm()
                self.psi = ansatz(closed=closed_ansatz,**kwargs)
            else:
                self.psi = ansatz(closed=closed_ansatz,**kwargs)
                self.rdm = rdm
                self.e0 = self.evaluate()
            self.ei = self.H.hf.e_tot
        elif self.H.model in ['fermion','fermi','fermionic']:
            self.No_as = self.H.No_as
            self.Ne_as = self.H.Ne_as
            self.alpha_mo = self.H.alpha_mo
            self.beta_mo  = self.H.beta_mo
            self._get_HF_rdm()
            self.e0 = self.hf_rdm.observable(self.H.matrix)+self.H._en_c
            self.ei = copy(self.e0)
            if use_initial:
                self.psi = copy(ansatz)
                self.rdm = rdm
                self.e0 = self.evaluate()
            else:
                #self.S = Operator()
                self.psi = ansatz(closed=closed_ansatz,**kwargs)
                # use input state
        else:
            print('Model: {}'.format(self.H.model))
            raise NotImplementedError
        sz = (self.H.Ne_alp-self.H.Ne_bet)*0.5
        self.comp_K2 = CompactRDM(
                order=2,
                alpha = self.alpha_mo['qubit'],
                beta  = self.beta_mo['qubit'],
                Ne=self.Ne_as,
                Sz=sz,
                rdm=Hamiltonian.K2)
        self.K2 = Hamiltonian.K2
        self.e0 = self.evaluate()

    def update(self,rdm):
        self.rdm = rdm


    def evaluate(self,rdm=None,trace=1):
        #  compact RDM representation here
        if type(rdm)==type(None):
            rdm = self.rdm
        if isinstance(rdm,type(CompactRDM())):
            # find trace 
            en = np.dot(rdm.rdm,self.comp_K2.rdm)
            #trace = rdm.trace()
        else:
            rdm.contract()
            en = rdm.observable(self.H.matrix)
        return en + self.H._en_c

    def analysis(self,rdm='default'):
        if rdm=='default':
            rdm =self.rdm
        if self.H.model in ['molecule','mol','molecular','fermionic']:
            rdm.get_spin_properties()
            print('Sz: {:.8f}'.format(np.real(rdm.sz)))
            print('S2: {:.8f}'.format(np.real(rdm.s2)))
            print('N:  {:.8f}'.format(np.real(rdm.trace())))
            print('Molecular Reduced Density Matrix: ')
            rdm.contract()
            if np.count_nonzero(rdm.rdm)>0:
                print(np.real(rdm.rdm))
            else:
                print(rdm.rdm)

            print('Eigenvalues of 2-RDM:')
            negative=False
            for n,i in enumerate(np.linalg.eigvalsh(rdm.rdm)):
                if abs(i)>1e-10:
                    print(i)
                    if i<0:
                        print('Negative eigenvalue!')
                        negative=True
                        print(i)
                        neg_ind = copy(n)
            rdm.expand()
            rdm1 = rdm.reduce_order()
            print('1-RDM: ')
            print(rdm1.rdm)
            print('Eigenvalues of 1-RDM:')
            for i in np.linalg.eigvalsh(rdm1.rdm):
                if abs(i)>1e-10:
                    print(i)

        else:
            print('Density matrix:')
            print(rdm.rdm)

    def _get_HF_rdm(self):
        sz = (self.H.Ne_alp-self.H.Ne_bet)*0.5
        s2 = {0:0,0.5:0.75,-0.5:0.75,1:2}
        #
        #self.hf_rdm = CompactRDM(
        #        order=2,
        self.hf_rdm = RDM(
                order=2,
                alpha = self.alpha_mo['qubit'],
                beta  = self.beta_mo['qubit'],
                rdm='hf',
                Ne=self.Ne_as,
                Sz=sz,
                S2=s2[sz],
                )
        compact=  CompactRDM(
                order=2,
                alpha = self.alpha_mo['qubit'],
                beta  = self.beta_mo['qubit'],
                Ne=self.Ne_as,
                Sz=sz,
                )
        self.e0 = np.real(self.evaluate(self.hf_rdm))
        #self.rdm = self.hf_rdm
        self.rdm = compact

