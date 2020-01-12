import numpy as np
from copy import deepcopy as copy
from collections import Counter
import sys
from functools import reduce
from hqca.core import *
from hqca.tools import *

class StorageACSE(Storage):
    '''
    Storage object for use in ACSE calculation. In general, needs only a 

    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,
            Hamiltonian=None,
            casci=False,
            use_initial=False,
            initial='hartree-fock',
            second_quant=False,
            **kwargs):
        self.H = Hamiltonian
        self.p = Hamiltonian.order
        try:
            self.r = self.H.No_tot*2 # spin orbitals
        except:
            pass
        self.ansatz = []
        self.use_initial=use_initial
        if self.H.model in ['mol','molecular','molecule']:
            self.No_as = self.H.No_as
            self.Ne_as = self.H.Ne_as
            self.alpha_mo = self.H.alpha_mo
            self.beta_mo  = self.H.beta_mo
            self._get_HF_rdm()
            if casci:
                self.get_FCI_rdm()
                self._set_overlap()
            self.ei = self.H.hf.e_tot
        elif self.H.model in ['sq','single-qubit']:
            if use_initial:
                # only the +, ++, +++ states are non-zero
                # what is the ordering like? 
                self.S = Operator(ops=[])
                if len(initial)==0:
                    pass
                elif not second_quant:
                    for p,c,a in initial:
                        temp = PauliOperator(
                                pauli=p,
                                coeff=c,
                                add=a)
                        self.S+= temp
                else:
                    for c,i,sq,a in initial:
                        temp = QubitOperator(
                                coeff=c,
                                indices=i,
                                sqOp=sq,
                                add=a
                                )
                        temp.generateOperators(Nq=1)
                        self.S+= temp.formOperator()
            else:
                self.rdm = qRDM(
                        order=1,
                        Nq=1,
                        )
                self.e0 = self.rdm.observable(self.H.matrix)+self.H._en_c
                self.ei = copy(self.e0)

        elif self.H.model in ['tq','two-qubit']:
            if use_initial:
                # only the +, ++, +++ states are non-zero
                # what is the ordering like? 
                self.S = Operator(ops=[])
                if len(initial)==0:
                    pass
                elif not second_quant:
                    for p,c,a in initial:
                        temp = PauliOperator(
                                pauli=p,
                                coeff=c,
                                add=a)
                        self.S+= temp
                else:
                    for c,i,sq in initial:
                        temp = QubitOperator(
                                coeff=c,
                                indices=i,
                                sqOp=sq
                                )
                        temp.generateOperators(Nq=1)
                        self.S+= temp.formOperator()
            else:
                self.rdm = qRDM(
                        order=2,
                        Nq=2,
                        )
                self.e0 = self.rdm.observable(self.H.matrix)+self.H._en_c
                print('Current energy: {}'.format(self.e0))
                self.ei = copy(self.e0)
        else:
            print('Model: {}'.format(self.H.model))
            sys.exit('Specify model initialization.')


    def update(self,rdm):
        self.rdm = rdm

    def evaluate(self,rdm):
        rdm.contract()
        en = rdm.observable(self.H.matrix)
        return en + self.H._en_c

    def analysis(self):
        #print('  --  --  --  --  --  --  -- ')
        #print('--  --  --  --  --  --  --  --')
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
            #print(np.linalg.eigvalsh(self.rdm.rdm))
            self.rdm.expand()
        else:
            print('Density matrix:')
            print(self.rdm.rdm)

    def _set_overlap(self):
        self.d_hf_fci =  self.hf_rdm2.get_overlap(self.fci_rdm2)
        print('Distance between HF, FCI: {:.8f}'.format(
            np.real(self.d_hf_fci)))

    def _get_HF_rdm(self):
        self.hf_rdm = RDM(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                )
        self.e0 = np.real(self.evaluate(self.hf_rdm))
        self.rdm = copy(self.hf_rdm)

    def _get_FCI_rdm(self):
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
        self.fci_rdm2 = RDM(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='given',
                rdm = fci_rdm2,
                Ne=self.Ne_as,
                )
        self.fci_rdm2.contract()
