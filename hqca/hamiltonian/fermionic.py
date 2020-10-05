from hqca.core import *
from functools import reduce
import sys
from hqca.tools import *
import numpy as np
from pyscf import gto,mcscf,scf
import timeit
from copy import deepcopy as copy


class FermionicHamiltonian(Hamiltonian):
    def __init__(self,
            proxy_mol,
            ints_1e,
            ints_2e,
            ints_spatial=True,
            transform=None,
            int_thresh=1e-10,
            Ne_active_space='default',
            No_active_space='default',
            integral_basis='hf',
            generate_operators=True,
            normalize=True,
            verbose=True,
            en_con=None,
            en_fin=None,
            ):
        if verbose:
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('      -- HAMILTONIAN --  ')
            print('-- -- -- -- -- -- -- -- -- -- --')
        self.verbose = verbose
        self._model='fermionic'
        self.real=True
        self.imag=False
        self.C = np.identity(ints_1e.shape[0])
        self._order = 2
        if Ne_active_space=='default':
            self.Ne_as = proxy_mol.nelec[0]+proxy_mol.nelec[1]
        else:
            self.Ne_as = int(Ne_active_space)
        self.Ne_tot = proxy_mol.nelec[0]+proxy_mol.nelec[1]
        self.Ne_core = self.Ne_tot - self.Ne_as
        self.Ne_alp = proxy_mol.nelec[0]-self.Ne_core//2
        self.Ne_bet = proxy_mol.nelec[1]-self.Ne_core//2
        if No_active_space=='default':
            self.No_as = self.C.shape[0]
        else:
            self.No_as = int(No_active_space)
        self.No_tot = self.C.shape[0]
        self.r = 2*self.No_as
        self._generate_spin2spac_mapping()
        self.spin = proxy_mol.spin
        self.norm = normalize
        self._generate_active_space()
        if type(en_con)==type(None):
            self._en_c = 0 
        else:
            self._en_c = en_con
        if type(en_fin)==type(None):
            self.ef = 0
        else:
            self.ef = en_fin
        if ints_spatial:
            # transforming spatial orbitals to spin :) 
            if verbose:
                print('Transforming 1e integrals...')
            self.ints_1e = generate_spin_1ei(
                    ints_1e,
                    self.C.T,
                    self.C.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
            if verbose:
                print('Transforming 2e integrals...')
            self.ints_2e = generate_spin_2ei_phys(
                    ints_2e,
                    self.C.T,
                    self.C.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
            if verbose:
                print('Done!')
            self.K2 = np.zeros((self.r,self.r,self.r,self.r))
            if self.verbose:
                print('Transforming molecular integrals...')
            for i in range(0,self.r):
                for j in range(0,self.r):
                    temp = 0.5*self.ints_2e[i,:,j,:]
                    for k in range(0,self.r):
                        temp[k,k]+= (1/(self.Ne_tot-1))*self.ints_1e[i,j]
                    self.K2[i,:,j,:]+= temp[:,:]
        if self.verbose:
            print('... Done!')
        self._matrix = contract(self.K2)
        self._transform = transform
        if generate_operators:
            self._build_operator(int_thresh)
        else:
            self._qubOp = None
            self._ferOp = None
        if self.verbose:
            print('\n\n')

    @property
    def order(self):
        return self._order

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self,a):
        self._mapping = a

    @property
    def qubit_operator(self):
        return self._qubOp

    @qubit_operator.setter
    def qubit_operator(self,b):
        self._qubOp = b

    @property
    def fermi_operator(self):
        return self._ferOp

    @fermi_operator.setter
    def fermi_operator(self,b):
        self._ferOp = b

    @property
    def matrix(self):
        return self._matrix

    def _generate_active_space(self,
            spin_mapping='default',
            **kw
            ):
        '''
        Note, all orb references are in spatial orbitals. 
        '''
        self.alpha_mo={
                'inactive':[],
                'active':[],
                'virtual':[],
                'qc':[]
                }
        self.beta_mo={
                'inactive':[],
                'active':[],
                'virtual':[],
                'qc':[]
                }
        self.Ne_ia = self.Ne_tot-self.Ne_as
        self.No_ia = self.Ne_ia//2
        self.spin = spin_mapping
        self.No_v  = self.No_tot-self.No_ia-self.No_as
        if self.Ne_ia%2==1:
            raise(SpinError)
        if self.Ne_ia>0:
            self.active_space_calc='CASSCF'
        ind=0
        for i in range(0,self.No_ia):
            self.alpha_mo['inactive'].append(ind)
            ind+=1
        for i in range(0,self.No_as):
            self.alpha_mo['active'].append(ind)
            ind+=1
        for i in range(0,self.No_v):
            self.alpha_mo['virtual'].append(ind)
            ind+=1
        for i in range(0,self.No_ia):
            self.beta_mo['inactive'].append(ind)
            ind+=1
        for i in range(0,self.No_as):
            self.beta_mo['active'].append(ind)
            ind+=1
        for i in range(0,self.No_v):
            self.beta_mo['virtual'].append(ind)
            ind+=1

    def _generate_spin2spac_mapping(self):
        self.s2s = {}
        for i in range(0,self.No_tot):
            self.s2s[i]=i
        for i in range(self.No_tot,2*self.No_tot):
            self.s2s[i]=i-self.No_tot


    def build_separable_operator(self,
            ordering='default',  #ordering of H to use
            threshold=1e-5,   #cut off for ints
            specific_grouping=[], #not used for most
            ):
        '''
        takes the self._qubOp representation and generates
        input: keywords related to modifying self._qubOp

        output: generates self._qubOp_sep, which is a list of separated
        operators 
        '''
        self._qubOp_sep = []
        if ordering in ['given','specified']:
            for group in specific_grouping:
                new = Operator()
                for item in group:
                    for op in self._qubOp:
                        if op.s==item:
                            new+= copy(op)
                            break
                self._qubOp_sep.append(new)


    def _build_operator(self,int_thresh=1e-10):
        if self.norm:
            Cnorm = np.max(np.abs(self.K2))/(np.pi)
        else:
            Cnorm=1
        if self.verbose:
            print('Normalization: {}'.format(Cnorm))
            print('Time: ')
        t1 = timeit.default_timer()
        alp = self.alpha_mo['active']
        bet = self.beta_mo['active']
        o2q = {}
        for i in range(self.No_as):
            o2q[alp[i]]=i
            o2q[bet[i]]=i+self.No_as
        qubOp = Operator()
        ferOp = Operator()
        # 1e terms
        for p in alp+bet:
            P = o2q[p]
            for q in alp+bet:
                Q = o2q[q]
                if abs(self.ints_1e[p,q])<=int_thresh:
                    continue
                newOp = FermiString(
                        coeff=self.ints_1e[p,q]/Cnorm,
                        indices=[P,Q],
                        ops='+-',
                        N = len(alp+bet),
                        )
                ferOp+= newOp
        t2 = timeit.default_timer()
        if self.verbose:
            print('1e terms: {}'.format(t2-t1))

        # starting 2 electron terms
        for p in alp+bet:
            P = o2q[p]
            for r in alp+bet:
                R = o2q[r]
                if p==r:
                    continue
                i1 = (p==r)
                for s in alp+bet:
                    S = o2q[s]
                    i2,i3 = (s==p),(s==r)
                    if i1+i2+i3==3:
                        continue
                    for q in alp+bet:
                        Q = o2q[q]
                        i4,i5,i6 = (q==p),(q==r),(q==s)
                        if i1+i2+i3+i4+i5+i6>=3:
                            continue
                        if q==s:
                            continue
                        if abs(self.ints_2e[p,r,q,s])<=int_thresh:
                            continue
                        newOp = FermiString(
                                coeff=0.5*self.ints_2e[p,r,q,s]/Cnorm,
                                indices=[P,R,S,Q],
                                ops='++--',
                                N = len(alp+bet),
                                )
                        ferOp+= newOp
        new = ferOp.transform(self._transform)
        qubOp = Operator()
        for i in new:
            if abs(i.c)>int_thresh:
                qubOp+= i

        self._qubOp = qubOp
        self._ferOp = ferOp
        t3 = timeit.default_timer()
        if self.verbose:
            print('2e terms: {}'.format(t3-t2))
            print('-------------------')
            print('Fermionic Hamiltonian')
            print(ferOp)
            print('Hamiltonian in Pauli Basis:')
            print(qubOp)
            print('-------------------')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,mod):
        self._model = mod
