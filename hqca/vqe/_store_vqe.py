import numpy as np
from copy import deepcopy as copy
import sys
from functools import reduce
from hqca.core import *
from hqca.tools import * 
from hqca.operators import *

class StorageVQE(Storage):
    '''
    '''
    def __init__(self,
            Hamiltonian=None,
            casci=False,
            use_initial=False,
            initial='hf',
            second_quant=False,
            **kwargs):
        self.H = Hamiltonian
        self.p = Hamiltonian.order
        try:
            self.r = self.H.No_tot*2 # spin orbitals
        except:
            pass
        self.use_initial=use_initial
        self.initial = initial
        if self.H.model in ['mol','molecular','molecule','fermionic']:
            self.No_as = self.H.No_as
            self.Ne_as = self.H.Ne_as
            self.alpha_mo = self.H.alpha_mo
            self.beta_mo  = self.H.beta_mo
            self._get_HF_rdm()
            if casci:
                self.get_FCI_rdm()
                self._set_overlap()
            try:
                self.ei = self.H.hf.e_tot
            except:
                self.ei = 0
            self.T = Operator()
        else:
            sys.exit('Error in self.H.model in StorageVQE')

    def update(self,rdm):
        self.rdm = rdm

    def evaluate(self,rdm):
        rdm.contract()
        en = rdm.observable(self.H.matrix)
        return en + self.H._en_c

    def _get_HF_rdm(self):
        sz = (self.H.Ne_alp-self.H.Ne_bet)*0.5
        s2 = {0:0,0.5:0.75,-0.5:0.75,1:2}
        self.hf_rdm = RDM(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                rdm='hf',
                Ne=self.Ne_as,
                Sz=sz,
                S2=s2[sz],
                )
        self.e0 = np.real(self.evaluate(self.hf_rdm))
        self.rdm = copy(self.hf_rdm)

    def analysis(self):
        if self.H.model in ['molecule','mol','molecular']:
            self.rdm.get_spin_properties()
            print('Sz: {:.8f}'.format(np.real(self.rdm.sz)))
            print('S2: {:.8f}'.format(np.real(self.rdm.s2)))
            self.rdm.contract()
            print('Molecular Reduced Density Matrix: ')
            print(np.real(self.rdm.rdm))
            print('Eigenvalues of density matrix:')
            for i in np.linalg.eigvalsh(self.rdm.rdm):
                if abs(i)>1e-10:
                    print(i)
            self.rdm.expand()
        else:
            print('Density matrix:')
            print(self.rdm.rdm)
