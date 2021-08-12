from hqca.core import *
from functools import reduce
import sys
from hqca.tools import *
from hqca.operators import *
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
            print_transformed=True,
            ):
        if verbose:
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('      -- HAMILTONIAN --  ')
            print('-- -- -- -- -- -- -- -- -- -- --')
        self.verbose = verbose
        self._model='fermionic'
        self.real=True
        self.imag=False
        self._order = 2
        self._transform = transform
        self._print_transformed=print_transformed
        if type(transform)==type(None):
            raise HamiltonianError('Need to specify transform for Hamiltonian.')
        if Ne_active_space=='default':
            self.Ne_as = proxy_mol.nelec[0]+proxy_mol.nelec[1]
        else:
            self.Ne_as = int(Ne_active_space)
        self.Ne_tot = proxy_mol.nelec[0]+proxy_mol.nelec[1]
        self.Ne_core = self.Ne_tot - self.Ne_as
        self.Ne_alp = proxy_mol.nelec[0]-self.Ne_core//2
        self.Ne_bet = proxy_mol.nelec[1]-self.Ne_core//2
        self.No_core = self.Ne_core//2
        if No_active_space=='default':
            self.C = np.identity(ints_1e.shape[0])
            self.No_as = self.C.shape[0]
        else:
            self.C = np.identity(No_active_space)
            self.No_as = int(No_active_space)
        self._core = [i for i in range(self.No_core)]
        self._active = [i+self.No_core for i in range(self.No_as)]
        self.No_tot = self.C.shape[0]
        self.r = 2*self.No_as
        self._generate_spin2spac_mapping()
        self.spin = proxy_mol.spin
        self.norm = normalize
        self._generate_active_space()
        self.ints_1e_given = ints_1e
        self.ints_2e_given = ints_2e
        if type(en_con)==type(None):
            self._en_c = 0
        else:
            self._en_c = en_con
        if type(en_fin)==type(None):
            self.ef = 0
        else:
            self.ef = en_fin
        self._gen_operators = generate_operators
        self._int_thresh = int_thresh
        self._update_ints(self.C,self.C)

    def _update_ints(self,mo_coeff_a,mo_coeff_b):
        self.mo_a =  mo_coeff_a
        self.mo_b = mo_coeff_b
        self.ints_1e = generate_spin_1ei(
                self.ints_1e_given.copy(),
                mo_coeff_a.T,
                mo_coeff_b.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s
                )
        self.ints_2e = generate_spin_2ei_phys(
                self.ints_2e_given.copy(),
                mo_coeff_a.T,
                mo_coeff_b.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s
                )
        self._build_K2()


    def _build_K2(self):
        self.K2 = np.zeros((self.r, self.r, self.r, self.r))
        self.K2+= self.ints_2e*0.5
        for i in range(0,self.r):
            for j in range(0,self.r):
                for k in range(self.r):
                    self.K2[i, k, j, k] += self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
                    self.K2[k, i, k, j] += self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
                    self.K2[i, k, k, j] -= self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
                    self.K2[k, i, j, k] -= self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
        self._matrix = contract(self.K2)
        if self.verbose:
            print('Core energy', self._en_c)
        if self._gen_operators:
            self._build_operator(self._int_thresh)
        else:
            self._qubOp = None
            self._ferOp = None

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
        self.No_v = self.No_tot - self.No_core-self.No_as
        self.alpha_mo={
                'inactive':[i for i in range(self.No_core)],
                'active':[i+self.No_core for i in range(self.No_as)],
                'virtual':[self.No_tot-self.No_v+i for i in range(self.No_v)],
                'qubit':[i for i in range(self.No_as)]
                }
        self.beta_mo={
                'inactive':[i+self.No_tot for i in range(self.No_core)],
                'active':[i+self.No_core+self.No_tot for i in range(self.No_as)],
                'virtual':[i+2*self.No_tot-self.No_v for i in range(self.No_v)],
                'qubit':[i+self.No_as for i in range(self.No_as)]
                }
        #print(self.alpha_mo)
        #print(self.beta_mo)
        self.spin = spin_mapping
        self.No_v  = self.No_tot-self.No_core-self.No_as

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
        if ordering in ['given','specified','default']:
            for group in specific_grouping:
                new = Operator()
                for item in group:
                    for op in self._qubOp:
                        if op.s==item:
                            new+= copy(op)
                            break
                self._qubOp_sep.append(new)


    def _build_operator(self,int_thresh=1e-14,compact=False):
        if self.verbose:
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
        #
        #
        for p in alp+bet:
            P = o2q[p]
            for q in alp+bet:
                Q = o2q[q]
                if abs(self.ints_1e[p,q])<=int_thresh:
                    continue
                newOp = FermiString(
                        N=len(alp+bet),
                        coeff=self.ints_1e[p,q],
                        indices=[P,Q],
                        ops='+-',
                        )
                ferOp+= newOp
        t2 = timeit.default_timer()
        if self.verbose:
            print('1e terms: {}'.format(t2-t1))
        t_transform = 0
        n=0
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
                        #if abs(self.K2[P,R,Q,S])<=int_thresh:
                        #    continue
                        newOp = FermiString(
                                N=len(alp+bet),
                                coeff=0.5*self.ints_2e[p,r,q,s],
                                #coeff=self.K2[P,R,Q,S],
                                indices=[P,R,S,Q],
                                ops='++--',
                                )
                        ferOp+= newOp
                        #t0 = dt()
                        #qubOp+= self._transform(newOp)
                        #t_transform+= dt()-t0
                        #n+=1
        t3 = timeit.default_timer()
        if self.verbose:
            print('2e terms: {}'.format(t3-t2))
        new = ferOp.transform(self._transform)
        qubOp = Operator()
        for i in new:
            if abs(i.c)>int_thresh:
                qubOp+= i
        self._qubOp = qubOp
        self._ferOp = ferOp
        t4 = timeit.default_timer()
        if self.verbose:
            print('2e transform: {}'.format(t4-t3))
        #print('2e transform: {}'.format(t_transform))
        if self.verbose and self._print_transformed:
            print('2e terms: {}'.format(t3-t2))
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('Second Quantized Hamiltonian')
            print(ferOp)
            print('Pauli String Hamiltonian:')
            print(qubOp)
            print('-- -- -- -- -- -- -- -- -- -- --')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,mod):
        self._model = mod
