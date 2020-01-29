from hqca.core import *
from functools import reduce
import sys
from hqca.tools import *
import numpy as np
from pyscf import gto,mcscf,scf


class MolecularHamiltonian(Hamiltonian):
    def __init__(self,
            mol,
            mapping='jordan-wigner',
            int_thresh=1e-10,
            Ne_active_space='default',
            No_active_space='default',
            orbitals='hf',
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
            self.Ne_as = int(Ne_as)
        self.Ne_tot = mol.nelec[0]+mol.nelec[1]
        self.Ne_alp = mol.nelec[0]
        self.Ne_bet = mol.nelec[1]
        if self.verbose:
            print('Hartree-Fock Energy: {:.8f}'.format(float(self.hf.e_tot)))
        if No_active_space=='default':
            self.No_as = self.C.shape[0]
        else:
            self.No_as = int(No_as)
        self.No_tot = self.C.shape[0]
        self.r = 2*self.No_tot
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
        if self._mo_basis in ['default','hf']:
            self.ints_1e = generate_spin_1ei(
                    self.ints_1e_ao.copy(),
                    self.C.T,
                    self.C.T,
                    self.alpha_mo,
                    self.beta_mo,
                    region='full',
                    spin2spac=self.s2s
                    )
            self.ints_2e = generate_spin_2ei(
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
        self._build_operator(int_thresh)

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
        alp = self.alpha_mo['active']
        bet = self.beta_mo['active']
        qubOp = Operator()
        ferOp = Operator()
        for p in alp+bet:
            for q in alp+bet:
                #if p>q:
                #    continue
                if abs(self.ints_1e[p,q])<=int_thresh:
                    continue
                #print('---',p,q,self.ints_1e[p,q])
                #print(p,q,self.ints_1e[p,q])
                newOp = FermionicOperator(
                        coeff=self.ints_1e[p,q],
                        indices=[p,q],
                        sqOp='+-',
                        antisymmetric=True,
                        add=True
                        )
                newOp.generateOperators(2*self.No_tot)
                ferOp+= newOp
                qubOp+= newOp.formOperator()
        for p in alp+bet:
            for r in alp+bet:
                if p==r:
                    continue
                i1 = (p==r)
                for s in alp+bet:
                    i2,i3 = (s==p),(s==r)
                    if i1+i2+i3==3:
                        continue
                    for q in alp+bet:
                        i4,i5,i6 = (q==p),(q==r),(q==s)
                        if i1+i2+i3+i4+i5+i6>=3:
                            continue
                        if q==s:
                            continue
                        if abs(self.ints_2e[p,r,q,s])<=int_thresh:
                        #if abs(self.K2[p,r,q,s])<=int_thresh:
                            continue
                        #print('--',p,r,s,q,self.K2[p,r,q,s])
                        #print('--',p,r,s,q,self.ints_2e[p,r,q,s])
                        newOp = FermionicOperator(
                                coeff=0.5*self.ints_2e[p,r,q,s],
                                #coeff=self.K2[p,r,q,s],
                                indices=[p,r,s,q],
                                sqOp='++--',
                                antisymmetric=True,
                                add=True
                                )
                        #print(newOp)
                        newOp.generateOperators(2*self.No_tot)
                        #for pa,c in zip(newOp.pPauli,newOp.pCoeff):
                        #    if pa=='XZYIII':
                        #        print(pa,c,p,r,s,q,self.ints_2e[p,r,q,s])
                        ferOp+= newOp
                        qubOp+= newOp.formOperator()
        qubOp.clean()
        self._qubOp = qubOp
        self._ferOp = ferOp
        if self.verbose:
            print(qubOp)


    def _old_build_operator(self,int_thresh=1e-14):
        '''
        Based on integrals, builds operator
        '''
        alp = self.alpha_mo['active']
        bet = self.beta_mo['active']
        blocks = [
                [alp,bet],
                [alp,bet]
                ]
        spins = [['a','b'],['a','b']]
        hold_op = {
                'no':[],
                'se':[],
                'nn':[],
                'ne':[],
                'de':[]}
        for ze in range(len(blocks[0])):
            for p in blocks[0][ze]:
                for q in blocks[1][ze]:
                    if p<q:
                        continue
                    if abs(self.ints_1e[p,q])<int_thresh:
                        continue
                    spin = ''.join([spins[0][ze],spins[1][ze]])
                    newOp = FermionicOperator(
                            coeff=self.ints_1e[p,q],
                            indices=[p,q],
                            sqOp='+-',
                            antisymmetric=False,
                            spin=spin,
                            )
                    if p==q:
                        hold_op['no'].append(newOp)
                    else:
                        hold_op['se'].append(newOp)
        for p in alp+bet:
            for q in alp+bet: 
                if p==q:
                    continue
                term_pqqp = self.ints_2e[p,q,p,q]
                term_pqqp-= self.ints_2e[p,q,q,p]
                if abs(term_pqqp)>int_thresh:
                    c1 =  (p in self.alpha_mo['active'])
                    c2 =  (q in self.alpha_mo['active'])
                    spin = '{}{}{}{}'.format(
                            c1*'a'+(1-c1)*'b',
                            c2*'a'+(1-c2)*'b',
                            c2*'a'+(1-c2)*'b',
                            c1*'a'+(1-c1)*'b',
                            )
                    newOp = FermionicOperator(
                            coeff=term_pqqp,
                            indices=[p,q,q,p],
                            sqOp='++--',
                            antisymmetric=False,
                            spin=spin,
                            )
                    hold_op['nn'].append(newOp)
        #
        # prrq operators
        #
        for p in alp:
            for q in alp:
                for r in alp+bet:
                    if r==p or r==q:
                        continue
                    term_pqrr = self.ints_2e[p,r,q,r]
                    c2 =  (r in alp)
                    spin = '{}{}{}{}'.format(
                            'a',c2*'a'+(1-c2)*'b',
                            c2*'a'+(1-c2)*'b','a')
                    if abs(term_pqrr)>int_thresh:
                        newOp = FermionicOperator(
                                coeff=term_pqrr,
                                indices=[p,r,r,q],
                                sqOp='++--',
                                spin=spin,
                                antisymmetric=False,
                                )
                        hold_op['ne'].append(newOp)
        for p in bet:
            for q in bet:
                for r in alp+bet:
                    if r==p or r==q:
                        continue
                    term_pqrr = self.ints_2e[p,r,q,r]
                    c2 =  (r in alp)
                    spin = '{}{}{}{}'.format(
                            'b',c2*'a'+(1-c2)*'b',
                            c2*'a'+(1-c2)*'b','b')
                    if abs(term_pqrr)>int_thresh:
                        newOp = FermionicOperator(
                                coeff=term_pqrr,
                                indices=[p,r,r,q],
                                sqOp='++--',
                                spin=spin,
                                antisymmetric=False,
                                )
                        hold_op['ne'].append(newOp)
        # prsq operators
        for p in alp:
            for r in alp:
                if p==r:
                    continue
                for s in alp:
                    if r==s or s==p:
                        continue
                    for q in alp:
                        if s==q or q==p or q==r:
                            continue
                        term1 = self.ints_2e[p,r,q,s]
                        if abs(term1)>int_thresh:
                            newOp = FermionicOperator(
                                    coeff=term1,
                                    indices=[p,r,s,q],
                                    sqOp='++--',
                                    spin='aaaa',
                                    antisymmetric=False,
                                    )
                            hold_op['de'].append(newOp)
        for p in bet:
            for r in bet:
                if p==r:
                    continue
                for s in bet:
                    if r==s or p==s:
                        continue
                    for q in bet:
                        if s==q or q==r or q==p:
                            continue
                        term1 = self.ints_2e[p,r,q,s]
                        if abs(term1)>int_thresh:
                            newOp = FermionicOperator(
                                    coeff=term1,
                                    indices=[p,r,s,q],
                                    sqOp='--++',
                                    antisymmetric=False,
                                    spin='bbbb',
                                    )
                            hold_op['de'].append(newOp)
        for p in alp:
            for r in bet:
                for s in bet:
                    if r==s:
                        continue
                    for q in alp:
                        if p==q:
                            continue
                        term1 = self.ints_2e[p,r,q,s]
                        if abs(term1)>int_thresh:
                            newOp = FermionicOperator(
                                    coeff=term1,
                                    indices=[p,r,s,q],
                                    sqOp='++--',
                                    spin='abba',
                                    antisymmetric=False,
                                    )
                            hold_op['de'].append(newOp)
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('      --  INTEGRALS --  ')
        print('-- -- -- -- -- -- -- -- -- -- --')
        ferOp = Operator()
        qubOp = Operator()
        name = {
                'ne':'Number excitations:',
                'nn':'Coulomb operators:',
                'de':'Double excitations:',
                'se':'Single excitations:',
                'no':'Number operators:'
                }
        for k,v in hold_op.items():
            print('{}'.format(name[k]))
            for item in v:
                print(item.qInd,item.qOp,item.qSp,item.ind,item.qCo)
        for key, item in hold_op.items():
            for op in item:
                #op.generateExponential(
                #        real=True,
                #        imag=False,
                #        Nq=2*self.No_as)
                op.generateOperators(Nq=2*self.No_as)
                #op.generateHermitianExcitationOperators(Nq=2*self.No_as)
            def simplify(tp,tc):
                done = False
                def check_duplicate(paulis):
                    for n in range(len(paulis)):
                        for m in range(n):
                            if paulis[n]==paulis[m]:
                                return m,n
                    return False,False
                while not done:
                    n,m = check_duplicate(tp)
                    if n==False and m==False:
                        done=True
                    else:
                        tc[n]+= tc.pop(m)
                        tp.pop(m)
                for i in reversed(range(len(tp))):
                    if abs(tc[i])<int_thresh:
                        tp.pop(i)
                        tc.pop(i)
                return tp,tc
            # 
            # now, start simplification in subgroups
            #
        pauli = []
        coeff = []
        tp,tc = [],[]
        qubOp+= hold_op['no']
        for op in hold_op['no']:
            ferOp+= op
            for p,c in zip(op.pPauli,op.pCoeff):
                tp.append(p)
                tc.append(c)
        for op in hold_op['nn']:
            ferOp+=op
            for p,c in zip(op.pPauli,op.pCoeff):
                tp.append(p)
                tc.append(c)
        tp,tc = simplify(tp,tc)
        pauli += tp[:]
        coeff += tc[:]
        op_list = []
        for item1 in hold_op['se']:
            op_list.append(item1)
        for n,item2 in enumerate(hold_op['ne']):
            op_list.append(item2)
        tp ,tc = [],[]
        for item in op_list:
            ferOp+=item
            for p,c in zip(item.pPauli,item.pCoeff):
                tp.append(p)
                tc.append(c)
        tp,tc = simplify(tp,tc)
        pauli += tp[:]
        coeff += tc[:]

        # simplification procedure for double excitations
        tp,tc = [],[]
        for item in hold_op['de']:
            ferOp+= item
            for p,c in zip(item.pPauli,item.pCoeff):
                tp.append(p)
                tc.append(c)
        tp,tc = simplify(tp,tc)
        pauli += tp[:]
        coeff += tc[:]

        #  # # # done
        new = Operator()
        print('-- -- -- -- -- -- -- -- -- -- --')
        for p,c in zip(pauli,coeff):
            new += PauliOperator(p,c.real)
            print('Term: {}, Value: {:.8f}'.format(p,c.real))
        print('-- -- -- -- -- -- -- -- -- -- --')
        self._qubOp = new
        self._ferOp = ferOp

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,mod):
        self._model = mod
