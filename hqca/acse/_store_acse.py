import numpy as np
from copy import deepcopy as copy
from collections import Counter
import sys
np.set_printoptions(linewidth=200)
from hqca.quantum.QuantumFunctions import QuantumStorage
from hqca.tools import Functions as fx
from hqca.tools import Chem as chem
from hqca.tools.RDM import RDMs
from functools import reduce
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver,UnitsType
from hqca.tools.Fermi import FermiOperator as Fermi
from hqca.tools import RDMFunctions as rdmf
from hqca.core import Storage

class ACSEStorage(Storage):
    '''
    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,
            Hamiltonian=None
            **kwargs):
        self.H = Hamiltonian
        self.r = self.molH.No_tot*2 # spin orbitals
        self.ansatz = []
        self.get_HF_rdm()
        self.get_FCI_rdm()
        self._set_overlap()

    def update(self,rdm):
        self.rdm = rdm

    def evaluate(self,rdm):
        rdm.contract()
        en = reduce(np.dot, (self.H._matrix,rdm.rdm)).trace()
        return en + self.H._en_c

    def analysis(self):
        print('  --  --  --  --  --  --  -- ')
        print('--  --  --  --  --  --  --  --')
        self.rdm.get_spin_properties()
        print('Sz: {:.8f}'.format(np.real(self.rdm2.sz)))
        print('S2: {:.8f}'.format(np.real(self.rdm2.s2)))
        ovlp = self.rdm2.get_overlap(self.fci_rdm2)
        print('Distance from FCI RDM: {:.8f}'.format(
            ovlp))
        print('Normalized distance: {:.8f}'.format(
            ovlp/self.d_hf_fci))


    def _set_overlap(self):
        self.d_hf_fci =  self.hf_rdm2.get_overlap(self.fci_rdm2)
        print('Distance between HF, FCI: {:.8f}'.format(
            np.real(self.d_hf_fci)))

    def get_HF_rdm(self):
        self.hf_rdm2 = RDMs(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                )
        self.e_init = self.evaluate(self.h2_rdm2)

    def get_FCI_rdm(self):
        d1,d2 = self.mc.fcisolver.make_rdm12s(
                self.mc.ci,self.No_as,self.Ne_as)
        fci_rdm2 = np.zeros((
            self.No_as*2,self.No_as*2,self.No_as*2,self.No_as*2))
        for i in self.alpha_mo['active']:
            for k in self.alpha_mo['active']:
                for l in self.alpha_mo['active']:
                    for j in self.alpha_mo['active']:
                        p,q = self.s2s[i],self.s2s[j]
                        r,s = self.s2s[k],self.s2s[l]
                        fci_rdm2[i,k,j,l] = d2[0][p,q,r,s]
        for i in self.alpha_mo['active']:
            for k in self.beta_mo['active']:
                for l in self.beta_mo['active']:
                    for j in self.alpha_mo['active']:
                        #if i>=j and k>=l:
                        #    continue
                        p,q = self.s2s[i],self.s2s[j]
                        r,s = self.s2s[k],self.s2s[l]
                        fci_rdm2[i,k,j,l]+= d2[1][p,q,r,s]
                        fci_rdm2[k,i,j,l]+= -d2[1][p,q,r,s]
                        fci_rdm2[i,k,l,j]+= -d2[1][p,q,r,s]
                        fci_rdm2[k,i,l,j]+= d2[1][p,q,r,s]

        for i in self.beta_mo['active']:
            for k in self.beta_mo['active']:
                for l in self.beta_mo['active']:
                    for j in self.beta_mo['active']:
                        p,q = self.s2s[i],self.s2s[j]
                        r,s = self.s2s[k],self.s2s[l]
                        fci_rdm2[i,k,j,l] = d2[2][p,q,r,s]
        self.fci_rdm2 = RDMs(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='given',
                rdm = fci_rdm2,
                Ne=self.Ne_as,
                )
        self.fci_rdm2.contract()
