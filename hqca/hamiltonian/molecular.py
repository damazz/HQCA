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
            compact_K2=False,
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
        self._compact = compact_K2
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
            print('Hartree-Fock Energy: {:.12f}'.format(float(self.hf.e_tot)))
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
                print('CASCI Energy: {:.12f}'.format(float(self.ef)))

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
        if self._gen_operators:
            self._build_operator(self._int_thresh)
        else:
            self._qubOp = None
            self._ferOp = None


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
                        #self.K2[i, k, j, k]+= self.ints_1e[i, j] / (2 * (self.Ne_tot - 1))
                        #self.K2[k, i, k, j]+= self.ints_1e[i, j] / (2 * (self.Ne_tot - 1))
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
        #ferOp += FermiString(
        #                s = 'i'*len(alp+bet),
        #                coeff=copy(self._en_c)
        #                )
        #self._en_c = 0 
        if self.verbose:
            print('2e terms: {}'.format(t3-t2))
        new = ferOp.transform(self._transform)
        qubOp = Operator()
        for i in new:
            if abs(i.c)>int_thresh:
                qubOp+= i
        qubOp.clean(1e-8)
        ferOp.clean(1e-8)
        self._qubOp = qubOp
        self._ferOp = ferOp
        t4 = timeit.default_timer()
        if self.verbose:
            print('2e transform: {}'.format(t4-t3))
        print('2e transform: {}'.format(t_transform))
        # adding identitiy
        try:
            lq = len(next(iter(qubOp)).s)
        except StopIteration:
            fer = Operator()+FermiString(s='i'*len(alp+bet),coeff=1)
            qub = fer.transform(self._transform)
            lq = len(next(iter(qub)).s)
        iden = PauliString(coeff=copy(self._en_c),pauli='I'*lq)
        qubOp += iden
        #if self.verbose and self._print_transformed:
        print('2e terms: {}'.format(t3-t2))
        print('-- -- -- -- -- -- -- -- -- -- --')
        #    print('Second Quantized Hamiltonian')
        #    print(ferOp)
        #    print('Pauli String Hamiltonian:')
        #    print(qubOp)
        #    print('-- -- -- -- -- -- -- -- -- -- --')



    def pivoted_chol(self, M='max',err_tol = 1e-6):
        """
    #  pivoted_chol.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.14.2020
    
    ## A pivoted cholesky function for kernel functions
        A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive semi-definite operator. Only diagonal elements and select rows of that operator's matrix represenation are required.
        get_diag - A function which takes no arguments and returns the diagonal of the matrix when called.
        get_row - A function which takes 1 integer argument and returns the desired row (zero indexed).
        M - The maximum rank of the approximate decomposition; an integer.
        err_tol - The maximum error tolerance, that is difference between the approximate decomposition and true matrix, allowed. Note that this is in the Trace norm, not the spectral or frobenius norm.
        Returns: R, an upper triangular matrix of column dimension equal to the target matrix. It's row dimension will be at most M, but may be less if the termination condition was acceptably low error rather than max iters reached.
        """
        #temp = 1e-8 *np.identity(self.No_as**2)
        ints_2e = rotate_spatial_ei2(self.ints_2e_ao.copy(),self.C.T)
        V = np.reshape(ints_2e,(self.No_as**2,self.No_as**2))

        def get_diag():
            diag = np.diagonal(V)
            diag.flags.writeable=True
            return diag

        def get_row(i):
            return V[i,:]
        if M=='max':
            M = self.No_as**2

    
        d = np.copy(get_diag())
        N = len(d)
        n = int(N**(0.5))
    
        pi = list(range(N))
    
        R = np.zeros([M,N])
   
        err = np.sum(np.abs(d))
    
        m = 0
        while (m < M) and (err > err_tol):
    
            i = m + np.argmax([d[pi[j]] for j in range(m,N)])
    
            tmp = pi[m]
            pi[m] = pi[i]
            pi[i] = tmp

            R[m,pi[m]] = np.sqrt(d[pi[m]])
            Apim = get_row(pi[m])
            for i in range(m+1, N):
                if m > 0:
                    ip = np.inner(R[:m,pi[m]], R[:m,pi[i]])
                else:
                    ip = 0
                R[m,pi[i]] = (Apim[pi[i]] - ip) / R[m,pi[m]]
                d[pi[i]] -= pow(R[m,pi[i]],2)
    
            err = np.sum([d[pi[i]] for i in range(m+1,N)])
            m += 1
    
        R = R[:m,:]
        if self.verbose:
            print('Final rank: {}'.format(m))
            print('Error: {}'.format(err))
        self.ints_2e_chol = R

        self.chol_H1 = self.build_cholesky_operator_Hp()
        self.chol_H2 = []
        for i in range(m):
            Li = np.zeros((n,n))
            for j in range(N):
                a,b = j//n,j%n
                Li[a,b] = R[i,j]
            #print(Li)
            #print(np.linalg.eigvalsh(Li))
            Up,Np = self.build_cholesky_operator_Vp(Li)
            self.chol_H2.append([Up,Np])


    def build_cholesky_operator_Hp(self):
        alp = self.alpha_mo['active']
        bet = self.beta_mo['active']
        n= len(alp)
        H = Operator()
        for i in alp:
            for j in alp:
                for k in alp+bet:
                    if abs(self.ints_2e[i,k,j,k])<1e-8:
                        continue
                    H+= FermiString(
                        N=2*n,
                        coeff=0.5*self.ints_2e[i,k,j,k],
                        indices=[i,j],
                        ops='+-',
                        )
                    if abs(self.ints_2e[i,k,j,k])<1e-8:
                        continue
        for i in bet:
            for j in bet:
                for k in alp+bet:
                    H+= FermiString(
                        N=2*n,
                        coeff=0.5*self.ints_2e[i,k,j,k],
                        indices=[i,j],
                        ops='+-',
                        )
        for i in alp+bet:
            for j in alp+bet:
                if abs(self.ints_1e[i,j])<1e-8:
                    continue
                H+= FermiString(
                    N=2*n,
                    coeff=0.5*self.ints_1e[i,j],
                    indices=[i,j],
                    ops='+-',
                    )
        Hp = H.transform(self._transform)
        Hp.clean(1e-8)
        return Hp

    def build_cholesky_operator_Vp(self,Li):
        eigval, U = np.linalg.eig(Li)
        # QR decomposition
        done = False
        n = len(eigval)
        temp = np.copy(U)
        #print(U)
        Uf = Operator()
        for c in range(0,n):
            for r in reversed(range(c+1,n)):
                #print(r,c)
                if abs(temp[r,c])<1e-8:
                    # already 0 
                    continue
                elif abs(temp[r-1,c])<1e-8:
                    # swap circuit
                    theta = np.pi/2
                else:
                    tan  = temp[r,c]/temp[r-1,c]

                    theta = np.arctan(tan)
                givens = np.identity(n)
                givens[r,r]=np.cos(theta)
                givens[r-1,r-1]=np.cos(theta)
                givens[r-1,r]= + np.sin(theta)
                givens[r,r-1]= - np.sin(theta)

                temp = np.dot(givens,temp)
                Uf+= FermiString(
                        N=2*n, 
                        coeff=theta,
                        indices=[r,r-1],
                        ops='+-',
                        )
                Uf+= FermiString(
                        N=2*n, 
                        coeff=-theta,
                        indices=[r-1,r],
                        ops='+-',
                        )
                Uf+= FermiString(
                        N=2*n, 
                        coeff=theta,
                        indices=[r+n,r+n-1],
                        ops='+-',
                        )
                Uf+= FermiString(
                        N=2*n, 
                        coeff=-theta,
                        indices=[r+n-1,r+n],
                        ops='+-',
                        )
        Nf = Operator()
        for i in range(n):
            if abs(eigval[i])<1e-6:
                continue
            for j in range(n):
                if abs(eigval[j])<1e-6:
                    continue
                Nf+= FermiString( #aa
                        N=2*n,
                        coeff=eigval[i]*eigval[j]*0.5,
                        ops='+-+-',
                        indices=[i,i,j,j]
                        )
                Nf+= FermiString( #ab
                        N=2*n,
                        coeff=eigval[i]*eigval[j]*0.5,
                        ops='+-+-',
                        indices=[i+n,i+n,j,j]
                        )
                Nf+= FermiString( #ab
                        N=2*n,
                        coeff=eigval[i]*eigval[j]*0.5,
                        ops='+-+-',
                        indices=[i,i,j+n,j+n]
                        )
                Nf+= FermiString( #ab
                        N=2*n,
                        coeff=eigval[i]*eigval[j]*0.5,
                        ops='+-+-',
                        indices=[i+n,i+n,j+n,j+n]
                        )
        Up = Uf.transform(self._transform)
        Up.clean(1e-8)
        Np = Nf.transform(self._transform)
        Np.clean(1e-8)
        return Up, Np







    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,mod):
        self._model = mod

