from hqca.core import *
from functools import reduce
import sys
from hqca.tools import *
import numpy as np
from pyscf import gto,mcscf,scf
import timeit


class MolecularHamiltonian(Hamiltonian):
    def __init__(self,
            mol,
            mapping='jordan-wigner',
            kw_mapping={},
            int_thresh=1e-10,
            Ne_active_space='default',
            No_active_space='default',
            orbitals='hf',
            operators='calc',
            verbose=True
            ):
        if verbose:
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('      -- HAMILTONIAN --  ')
            print('-- -- -- -- -- -- -- -- -- -- --')
        self._mo_basis = orbitals
        self.verbose = verbose
        self.S = mol.intor('int1e_ovlp')
        self.T_1e = mol.intor('int1e_kin')
        self.V_1e = mol.intor('int1e_nuc')
        self.ints_1e_ao = self.V_1e+self.T_1e
        self._model='molecule'
        self.ints_2e_ao = mol.intor('int2e')
        self.real=True
        self.imag=False
        self.hf = scf.ROHF(mol)
        self.hf.kernel()
        self.hf.analyze()
        self.e0 = self.hf.e_tot
        self.C = self.hf.mo_coeff
        self.f = self.hf.get_fock()
        self._order = 2
        if Ne_active_space=='default':
            self.Ne_as = mol.nelec[0]+mol.nelec[1]
        else:
            self.Ne_as = int(Ne_active_space)
        self.Ne_tot = mol.nelec[0]+mol.nelec[1]
        self.Ne_core = self.Ne_tot - self.Ne_as
        self.Ne_alp = mol.nelec[0]-self.Ne_core//2
        self.Ne_bet = mol.nelec[1]-self.Ne_core//2
        if self.verbose:
            print('Hartree-Fock Energy: {:.8f}'.format(float(self.hf.e_tot)))
        if No_active_space=='default':
            self.No_as = self.C.shape[0]
        else:
            self.No_as = int(No_active_space)
        self.No_tot = self.C.shape[0]
        self.r = 2*self.No_as
        self._generate_spin2spac_mapping()
        if self.No_as<=4:
            self.mc = mcscf.CASCI(
                    self.hf,
                    self.No_as,
                    self.Ne_as)
            self.mc.kernel()
            self.ef  = self.mc.e_tot
            self.mc_coeff = self.mc.mo_coeff
        else:
            self.ef =  0
        if self.verbose:
            print('CASCI Energy: {:.8f}'.format(float(self.ef)))
        self.spin = mol.spin
        self.Ci = np.linalg.inv(self.C)
        self._generate_active_space()
        self._en_c = mol.energy_nuc()
        if self._mo_basis in ['default','hf','active','as']:
            if verbose:
                print('Transforming 1e integrals...')
            self.ints_1e = generate_spin_1ei(
                    self.ints_1e_ao.copy(),
                    self.C.T,
                    self.C.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
            if verbose:
                print('Transforming 2e integrals...')
            self.ints_2e = generate_spin_2ei(
                    self.ints_2e_ao.copy(),
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
            if self._mo_basis in ['active','as']:
                active = self.alpha_mo['active']+self.beta_mo['active']
                for i in range(0,self.r):
                    I = active[i]
                    for j in range(0,self.r):
                        J = active[j]
                        for k in range(0,self.r):
                            K = active[k]
                            self.K2[i,k,j,k]+=(
                                    1/(self.Ne_tot-1)
                                    )*self.ints_1e[I,J]
                            for l in range(0,self.r):
                                L = active[l]
                                self.K2[i,k,j,l]+= 0.5*self.ints_2e[I,K,J,L]
                print('Done!')
            else:
                for i in range(0,self.r):
                    for j in range(0,self.r):
                        temp = 0.5*self.ints_2e[i,:,j,:]
                        for k in range(0,self.r):
                            temp[k,k]+= (1/(self.Ne_tot-1))*self.ints_1e[i,j]
                        self.K2[i,:,j,:]+= temp[:,:]
        elif self._mo_basis=='no':
            if self.verbose:
                print('Obtaining natural orbitals.')
            d1 = self.mc.fcisolver.make_rdm1s(
                    self.mc.ci,
                    self.No_as,
                    self.Ne_as,
                    )
            def reorder(rdm1,orbit):
                '''
                Finds the transformation to obtain the Aufbau ordering 
                for spatial orbitals: the spatial orbitals according 
                to the eigenvalues of the 1-RDM (sometimes, 
                diagonalization procedure will swap the orbital 
                ordering). 
                '''
                ordered=False
                T = np.identity(orbit)
                for i in range(0,orbit):
                    for j in range(i+1,orbit):
                        if rdm1[i,i]>=rdm1[j,j]:
                            continue
                        else:
                            temp= np.identity(orbit)
                            temp[i,i] = 0 
                            temp[j,j] = 0
                            temp[i,j] = -1
                            temp[j,i] = 1
                            T = np.dot(temp,T)
                return T
            nocca, norba = np.linalg.eig(d1[0]) # diagonalize alpha
            noccb, norbb = np.linalg.eig(d1[1]) # diagonalize beta 
            # reorder according to eigenvalues for alpha, beta
            Ta = reorder(reduce(np.dot, (norba.T,d1[0],norba)),self.No_as)
            Tb = reorder(reduce(np.dot, (norbb.T,d1[1],norbb)),self.No_as)
            # generate proper 1-RDM in NO basis, alpha bet
            D1_a = reduce(np.dot, (Ta.T, norba.T, d1[0], norba, Ta))
            D1_b = reduce(np.dot, (Tb.T, norbb.T, d1[1], norbb, Tb))
            # transformation from AO to NO for alpha, beta, using the 
            # provided HF solution as well
            ao2no_a = reduce(np.dot, (self.mc.mo_coeff, norba, Ta))
            ao2no_b = reduce(np.dot, (self.mc.mo_coeff, norbb, Tb))
            # Note, these are in (AO,NO) form, so they are: "ao to no"
            # important function, generates the full size 1e no (NOT 
            # in the spatial orbital basis, but in the spin basis) 
            self.ints_2e = generate_spin_2ei(
                    self.ints_2e_ao, 
                    ao2no_a.T, 
                    ao2no_b.T,
                    self.alpha_mo,
                    self.beta_mo,
                    spin2spac=self.s2s
                    )
            self.ints_1e = generate_spin_1ei(
                    self.ints_1e_ao,
                    ao2no_a.T,
                    ao2no_b.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
        elif self._mo_basis=='pyscf':
            self.ints_1e = generate_spin_1ei(
                    self.ints_1e_ao.copy(),
                    self.C.T,
                    self.C.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
            self.ints_2e = generate_spin_2ei_pyscf(
                    self.ints_2e_ao.copy(),
                    self.C.T,
                    self.C.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
            self.K2 = np.zeros((self.r,self.r,self.r,self.r))
            if self.verbose:
                print('Transforming molecular integrals...')
            for i in range(0,self.r):
                for j in range(0,self.r):
                    temp = 0.5*self.ints_2e[i,j,:,:]
                    for k in range(0,self.r):
                        temp[k,k]+= (1/(self.Ne_tot-1))*self.ints_1e[i,j]
                    self.K2[i,j,:,:]+= temp[:,:]
        if self.verbose:
            print('... Done!')
        self._matrix = contract(self.K2)
        self._model = 'molecular'
        self._mapping = mapping
        self._kw_mapping = kw_mapping
        if operators=='calc':
            self._build_operator(int_thresh)
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


    def _build_operator(self,int_thresh=1e-14):
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
                newOp = FermionicOperator(
                        coeff=self.ints_1e[p,q],
                        indices=[P,Q],
                        sqOp='+-',
                        antisymmetric=True,
                        add=True
                        )
                newOp.generateOperators(
                        Nq=2*self.No_as,
                        mapping=self._mapping,
                        **self._kw_mapping)
                ferOp+= newOp
                qubOp+= newOp.formOperator()
        # 2e terms
        t2 = timeit.default_timer()
        print('1e terms: {}'.format(t2-t1))
        for p in alp:
            P = o2q[p]
            for r in alp:
                R = o2q[r]
                if p==r:
                    continue
                i1 = (p==r)
                for s in alp:
                    S = o2q[s]
                    i2,i3 = (s==p),(s==r)
                    if i1+i2+i3==3:
                        continue
                    for q in alp:
                        Q = o2q[q]
                        i4,i5,i6 = (q==p),(q==r),(q==s)
                        if i1+i2+i3+i4+i5+i6>=3:
                            continue
                        if q==s:
                            continue
                        if abs(self.ints_2e[p,r,q,s])<=int_thresh:
                            continue
                        newOp = FermionicOperator(
                                coeff=0.5*self.ints_2e[p,r,q,s],
                                indices=[P,R,Q,S],
                                sqOp='++--',
                                antisymmetric=True,
                                add=True
                                )
                        newOp.generateOperators(
                                Nq=2*self.No_as,
                                mapping=self._mapping,
                                **self._kw_mapping
                                )
                        ferOp+= newOp
                        qubOp+= newOp.formOperator()
        for p in alp:
            P = o2q[p]
            for r in bet:
                R = o2q[r]
                if p==r:
                    continue
                i1 = (p==r)
                for s in bet:
                    S = o2q[s]
                    i2,i3 = (s==p),(s==r)
                    if i1+i2+i3==3:
                        continue
                    for q in alp:
                        Q = o2q[q]
                        i4,i5,i6 = (q==p),(q==r),(q==s)
                        if i1+i2+i3+i4+i5+i6>=3:
                            continue
                        if q==s:
                            continue
                        if abs(self.ints_2e[p,r,q,s])<=int_thresh:
                            continue
                        newOp = FermionicOperator(
                                coeff=0.5*self.ints_2e[p,r,q,s],
                                indices=[P,R,Q,S],
                                sqOp='++--',
                                antisymmetric=True,
                                add=True
                                )
                        newOp.generateOperators(
                                Nq=2*self.No_as,
                                mapping=self._mapping,
                                **self._kw_mapping
                                )
                        ferOp+= newOp
                        qubOp+= newOp.formOperator()
        for p in bet:
            P = o2q[p]
            for r in alp:
                R = o2q[r]
                if p==r:
                    continue
                i1 = (p==r)
                for s in alp:
                    S = o2q[s]
                    i2,i3 = (s==p),(s==r)
                    if i1+i2+i3==3:
                        continue
                    for q in bet:
                        Q = o2q[q]
                        i4,i5,i6 = (q==p),(q==r),(q==s)
                        if i1+i2+i3+i4+i5+i6>=3:
                            continue
                        if q==s:
                            continue
                        if abs(self.ints_2e[p,r,q,s])<=int_thresh:
                            continue
                        newOp = FermionicOperator(
                                coeff=0.5*self.ints_2e[p,r,q,s],
                                indices=[P,R,Q,S],
                                sqOp='++--',
                                antisymmetric=True,
                                add=True
                                )
                        newOp.generateOperators(
                                Nq=2*self.No_as,
                                mapping=self._mapping,
                                **self._kw_mapping
                                )
                        ferOp+= newOp
                        qubOp+= newOp.formOperator()
        for p in bet:
            P = o2q[p]
            for r in bet:
                R = o2q[r]
                if p==r:
                    continue
                i1 = (p==r)
                for s in bet:
                    S = o2q[s]
                    i2,i3 = (s==p),(s==r)
                    if i1+i2+i3==3:
                        continue
                    for q in bet:
                        Q = o2q[q]
                        i4,i5,i6 = (q==p),(q==r),(q==s)
                        if i1+i2+i3+i4+i5+i6>=3:
                            continue
                        if q==s:
                            continue
                        if abs(self.ints_2e[p,r,q,s])<=int_thresh:
                            continue
                        newOp = FermionicOperator(
                                coeff=0.5*self.ints_2e[p,r,q,s],
                                indices=[P,R,Q,S],
                                sqOp='++--',
                                antisymmetric=True,
                                add=True
                                )
                        newOp.generateOperators(
                                Nq=2*self.No_as,
                                mapping=self._mapping,
                                **self._kw_mapping
                                )
                        ferOp+= newOp
                        qubOp+= newOp.formOperator()
        qubOp.clean()
        self._qubOp = qubOp
        self._ferOp = ferOp
        t3 = timeit.default_timer()
        print('2e terms: {}'.format(t3-t2))
        if self.verbose:
            print('Hamiltonian in Pauli Basis:')
            print(qubOp)
            print('-------------------')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,mod):
        self._model = mod
