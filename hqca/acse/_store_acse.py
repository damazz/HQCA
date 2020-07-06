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
            if not use_initial:
                self._get_HF_rdm()
            else:
                self.S = copy(initial)
                self.S+= op
                for op in initial:
                    self.S+= op
                self._run_initial(**kwargs)
                # use input state
                sys.exit()
            if casci:
                self.get_FCI_rdm()
                self._set_overlap()
            self.ei = self.H.hf.e_tot
            self.S = Operator()
        elif self.H.model in ['sq','single-qubit']:
            if use_initial:
                # only the +, ++, +++ states are non-zero
                # what is the ordering like? 
                self.S = Operator(ops=[])
                if len(initial)==0:
                    pass
                elif not second_quant:
                    for p,c,a in initial:
                        temp = PauliString(
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
                # so...generate the S operator 
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
                self.S = Operator()
                # use input state
        elif self.H.model in ['tq','two-qubit']:
            if use_initial:
                # only the +, ++, +++ states are non-zero
                # what is the ordering like? 
                self.S = Operator(ops=[])
                if len(initial)==0:
                    pass
                elif not second_quant:
                    for p,c,a in initial:
                        temp = PauliString(
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
        #rdm.expand()
        #zed = np.nonzero(rdm.rdm)
        #for i,j,k,l in zip(zed[0],zed[1],zed[2],zed[3]):
        #    if rdm.rdm[i,j,k,l]>1e-3:
        #        print(i,j,k,l,rdm.rdm[i,j,k,l])
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
            print(np.real(rdm.rdm))

            print('Eigenvalues of 2-RDM:')
            negative=False
            for n,i in enumerate(np.linalg.eigvalsh(rdm.rdm)):
                if abs(i)>1e-10:
                    if i<0:
                        print('Negative eigenvalue!')
                        negative=True
                        print(i)
                        neg_ind = copy(n)
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

    def _set_overlap(self):
        self.d_hf_fci =  self.hf_rdm2.get_overlap(self.fci_rdm2)
        print('Distance between HF, FCI: {:.8f}'.format(
            np.real(self.d_hf_fci)))

    def _get_HF_rdm(self):
        self.hf_rdm = RDM(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='scf',
                Ne=self.Ne_as,
                S=self.H.Ne_alp-self.H.Ne_bet,
                )
        self.e0 = np.real(self.evaluate(self.hf_rdm))
        self.rdm = copy(self.hf_rdm)

    def _run_initial(self,op,Ins,**kwargs):
        pass

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
