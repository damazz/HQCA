import numpy as np
from copy import deepcopy as copy
from collections import Counter
import sys
from functools import reduce
from hqca.core import *
from hqca.tools import *
from hqca.cqe._pauli_ansatz import *

class StorageACSE(Storage):
    '''
    Storage object for use in ACSE calculation. In general, needs a Hamiltonian

    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,
            Hamiltonian=None,
            use_initial=False,
            initial='hartree-fock',
            second_quant=False,
            closed_ansatz=True,
            ansatz=None,
            **kwargs):
        self.H = Hamiltonian
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
            else:
                raise NotImplementedError
            self.ei = self.H.hf.e_tot
            self.S = ansatz(closed=closed_ansatz,**kwargs)
        elif self.H.model in ['fermion','fermi','fermionic']:
            self.No_as = self.H.No_as
            self.Ne_as = self.H.Ne_as
            self.alpha_mo = self.H.alpha_mo
            self.beta_mo  = self.H.beta_mo
            self._get_HF_rdm()
            self.e0 = self.hf_rdm.observable(self.H.matrix)+self.H._en_c
            self.ei = copy(self.e0)
            if use_initial:
                self.S = copy(initial)
            else:
                #self.S = Operator()
                self.S = Ansatz(closed=closed_ansatz,**kwargs)
                # use input state
        else:
            print('Model: {}'.format(self.H.model))
            raise NotImplementedError

    def update(self,rdm):
        self.rdm = rdm

    def evaluate(self,rdm):
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
            '''
            if negative:
                rdm.expand()
                zed = np.nonzero(rdm.rdm)
                alpalp = np.zeros(rdm.rdm.shape)
                alpbet = np.zeros(rdm.rdm.shape)
                betbet = np.zeros(rdm.rdm.shape)
                for i,j,k,l in zip(zed[0],zed[1],zed[2],zed[3]):
                    if abs(rdm.rdm[i,j,k,l])>1e-8:
                        print(i,j,k,l,rdm.rdm[i,j,k,l])
                        c1 = i in self.alpha_mo['active']
                        c2 = j in self.alpha_mo['active']
                        c3 = k in self.alpha_mo['active']
                        c4 = l in self.alpha_mo['active']
                        val = rdm.rdm[i,j,k,l]
                        if c1+c2+c3+c4==0:
                            betbet[i,j,k,l]=val
                        elif c1+c2+c3+c4==2:
                            alpbet[i,j,k,l]=val
                        elif c1+c2+c3+c4==4:
                            alpalp[i,j,k,l]=val
                rdm.contract()
                for lab,mat in zip(['aa','ab','bb'],[alpalp,alpbet,betbet]):
                    mat = np.reshape(mat,(rdm.rdm.shape))
                    print('EIGVALS OF {} matrix'.format(lab))
                    for n,i in enumerate(np.linalg.eigvalsh(mat)):
                        if abs(i)>1e-10:
                            if i<0:
                                print(i)
                #sys.exit('Negative eigenvalue?')
            '''
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
        self.hf_rdm = RDM(
                order=2,
                alpha = self.alpha_mo['qubit'],
                beta  = self.beta_mo['qubit'],
                rdm='hf',
                Ne=self.Ne_as,
                Sz=sz,
                S2=s2[sz],
                )
        self.e0 = np.real(self.evaluate(self.hf_rdm))
        self.rdm = copy(self.hf_rdm)

