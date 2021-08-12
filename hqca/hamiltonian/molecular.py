from hqca.core import *
from functools import reduce
import sys
from copy import deepcopy as copy
from hqca.tools import *
from hqca.operators import *
import numpy as np
from pyscf import gto,mcscf,scf
import timeit
from timeit import default_timer as dt

class MolecularHamiltonian(Hamiltonian):
    def __init__(self,
            mol,
            transform=None,
            int_thresh=1e-14,
            active_space=None,
            integral_basis='hf',
            generate_operators=True,
            verbose=True,
            en_c=None,
            solver='casci',
            print_transformed=True,
            ):
        if verbose:
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('      -- HAMILTONIAN --  ')
            print('-- -- -- -- -- -- -- -- -- -- --')
        self._mo_basis = integral_basis
        self.verbose = verbose
        self.S = mol.intor('int1e_ovlp')
        self.T_1e = mol.intor('int1e_kin')
        self.V_1e = mol.intor('int1e_nuc')
        self.ints_1e_ao = self.V_1e+self.T_1e
        self._model='molecule'
        self.ints_2e_ao = mol.intor('int2e')
        self.real=True
        self.imag=False
        self._print_transformed=print_transformed
        self._transform = transform
        if type(transform)==type(None):
            raise HamiltonianError('Need to specify transform for Hamiltonian.')
        self.hf = scf.ROHF(mol)
        self.hf.kernel()
        self.hf.analyze()
        self.e0 = self.hf.e_tot
        self.C = self.hf.mo_coeff
        self.f = self.hf.get_fock()
        self._order = 2
        self.mol = mol
        if isinstance(active_space,tuple) or isinstance(active_space,list):
            self._use_active_space = True
            self.Ne_as = int(active_space[0])
            self.No_as = int(active_space[1])
        else:
            self.No_as = self.C.shape[0]
            self._use_active_space = False
            self.Ne_as = mol.nelec[0]+mol.nelec[1]
        self.Ne_tot = mol.nelec[0]+mol.nelec[1]
        self.Ne_core = self.Ne_tot - self.Ne_as
        self.Ne_alp = mol.nelec[0]-self.Ne_core//2
        self.Ne_bet = mol.nelec[1]-self.Ne_core//2
        self.No_core = self.Ne_core//2
        self._core = [i for i in range(self.No_core)]
        self._active = [i+self.No_core for i in range(self.No_as)]
        if self.verbose:
            print('Hartree-Fock Energy: {:.8f}'.format(float(self.hf.e_tot)))
        self.No_tot = self.C.shape[0]
        self.r = 2*self.No_as
        if self.verbose:
            print('N electrons total: {}'.format(self.Ne_tot))
            print('N electrons active: {}'.format(self.Ne_as))
            print('N core orbitals: {}'.format(self.No_core))
            print('N active orbitals: {}'.format(self.No_as))

        self._generate_spin2spac_mapping()
        if solver in ['casci','fci','ci'] and self.No_as<=8:
            self.mc = mcscf.CASCI(
                    self.hf,
                    self.No_as,
                    self.Ne_as)
            self.mc.fcisolver.nroots = 4
            self.mc.kernel()
            if abs(self.mc.e_tot[1]-self.mc.e_tot[0])<0.01:
                print('Energy gaps:')
                for i in range(self.mc.fcisolver.nroots-1):
                    print(self.mc.e_tot[i+1]-self.mc.e_tot[i])
                print('Ground excited state energy gap less than 10 mH')
            self.ef  = self.mc.e_tot[0]
            self.mc_coeff = self.mc.mo_coeff
            if self.verbose:
                print('CASCI Energy: {:.8f}'.format(float(self.ef)))

        elif solver in ['casscf']:
            n_states = 2
            weights = np.ones(n_states)/n_states
            self.mc = mcscf.CASSCF(
                    self.hf,
                    self.No_as,
                    self.Ne_as).state_average_((1,0,0))
            self.mc.kernel()
            self.ef  = self.mc.e_tot
            self.mc_coeff = self.mc.mo_coeff
            if self.verbose:
                print('CASSCF Energy: {:.8f}'.format(float(self.ef)))
        else:
            self.ef = 0
        self.spin = mol.spin
        self.Ci = np.linalg.inv(self.C)
        self._generate_active_space()
        if self._mo_basis in ['default','hf']:
            mo_coeff_a,mo_coeff_b = copy(self.C),copy(self.C)
        elif self._mo_basis in ['no','natural','canonical']:
            print('Stating with natural orbital.... ')
            mo_coeff_a,mo_coeff_b = self.mc_coeff,self.mc_coeff
        if verbose:
            print('Transforming 1e integrals...')
        self.energy_nuclear = mol.energy_nuc()
        self._gen_operators = generate_operators
        self._int_thresh = int_thresh
        self._update_ints(mo_coeff_a,mo_coeff_b)

    def _update_ints(self,mo_coeff_a,mo_coeff_b):
        self.mo_a =  mo_coeff_a
        self.mo_b = mo_coeff_b
        self.ints_1e = generate_spin_1ei(
                self.ints_1e_ao.copy(),
                mo_coeff_a.T,
                mo_coeff_b.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s
                )
        self.ints_2e = generate_spin_2ei(
                self.ints_2e_ao.copy(),
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
        if self._use_active_space:
            active = self.alpha_mo['active']+self.beta_mo['active']
            core = self.alpha_mo['inactive'] + self.beta_mo['inactive']
            core_1e = np.zeros((self.r,self.r))
            for i in range(0,self.r):
                I = active[i]
                for j in range(0,self.r):
                    J = active[j]
                    core_1e[i,j]+= self.ints_1e[I,J]
                    for k in range(0,self.No_core*2):
                        # active-core electrons
                        K = core[k]
                        core_1e[i,j]+= self.ints_2e[I,K,J,K]
                        core_1e[i,j]-= self.ints_2e[I,K,K,J]
                    for k in range(0,self.r):
                        # active active 1e
                        K = active[k]
                        self.K2[i,k,j,k]+= core_1e[i,j]/(4*(self.Ne_as-1))
                        self.K2[k,i,k,j]+= core_1e[i,j]/(4*(self.Ne_as-1))
                        self.K2[i,k,k,j]-= core_1e[i,j]/(4*(self.Ne_as-1))
                        self.K2[k,i,j,k]-= core_1e[i,j]/(4*(self.Ne_as-1))
                        for l in range(0,self.r):
                            # active active 2e
                            L = active[l]
                            self.K2[i,k,j,l]+= 0.5*self.ints_2e[I,K,J,L]
        else:
            self.K2+= self.ints_2e*0.5
            for i in range(0,self.r):
                for j in range(0,self.r):
                    for k in range(self.r):
                        self.K2[i, k, j, k] += self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
                        self.K2[k, i, k, j] += self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
                        self.K2[i, k, k, j] -= self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
                        self.K2[k, i, j, k] -= self.ints_1e[i, j] / (4 * (self.Ne_tot - 1))
        self._matrix = contract(self.K2)
        self._model = 'molecular'
        if self._use_active_space:
            # need to trace over the other degrees of freedom...i.e.
            core_ab = self.alpha_mo['inactive']+self.beta_mo['inactive']
            E_core = 0
            for i in core_ab:
                E_core += self.ints_1e[i,i]
                for j in core_ab:
                    E_core+= 0.5*self.ints_2e[i,j,i,j]
                    E_core-= 0.5*self.ints_2e[i,j,j,i]
            self._en_c = E_core+self.energy_nuclear
        else:
            self._en_c = self.energy_nuclear
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


    def _update_integrals(self):
        pass

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
