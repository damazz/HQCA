import numpy as np
from scipy import linalg as la
np.set_printoptions(linewidth=300)
import sympy as sy
from hqca.tools._operator import *
from hqca.tools.quantum_strings import *


# generate the parity check matrix



class ParityCheckMatrix:
    def __init__(self,paulis,verbose=False):
        '''
        Given a Pauli operator (i.e., Operator class composed of Pauli strings),
        we are trying to construct the parity check matrix.  If hthere 

        '''
        self.verbose = verbose
        self.op = paulis
        if type(paulis)==type([]):
            self.N = len(paulis[0].s)
            self.l = len(paulis)
            self.E = np.zeros((self.l-1,2*self.N))
            iden = 'I'*self.N
            add = 0
            for i in range(self.l):
                if paulis[i].s==iden:
                    add = -1
                    continue
                else:
                    self.E[i+add,:]=self._pauli_to_check(paulis[i+add].s)
        else:
            self.N = len(paulis.op[0].s)
            self.l = len(paulis.op)
            self.E = np.zeros((self.l,2*self.N))
            iden = 'I'*self.N
            add = 0
            for i in range(self.l):
                if paulis.op[i].s==iden:
                    add = -1
                    continue
                else:
                    self.E[i+add,:]=self._pauli_to_check(paulis.op[i].s)
        if add==-1:
            self.E = self.E[:self.l-1,:]
            self.l-=1
        self.gaussian_elimination()
        #p,l,u  = la.lu(self.E)
        #print(p.shape)
        #print(l.shape)
        #print(u.shape)

    def _pauli_to_check(self,p):
        vec = np.zeros(2*self.N)
        for i in range(self.N):
            if p[i]=='Z' or p[i]=='Y':
                vec[i]=1
        for i in range(self.N,2*self.N):
            if p[i%self.N]=='X' or p[i%self.N]=='Y':
                vec[i]=1
        return vec

    def gaussian_elimination(self):
        def apply_h(vec):
            if not len(vec)==self.N*2 or not vec.shape[0]==self.N*2:
                print('Wrong vector length.')
            nvec = np.zeros(vec.shape[0])
            for i in range(vec.shape[0]):
                nvec[(i+self.N)%(2*self.N)]=vec[i]
            return nvec
        mat = np.copy(self.E)
        # n rows = self.l
        # n cols = self.N
        adjust=0
        dep = []
        ind = []
        for c in range(2*self.N):
            #
            if self.E[c+adjust,c]==1:
                pass
            else:
                done = False
                for r in range(c+1+adjust,self.l):
                    # for rows in the column range
                    if self.E[r,c]==1:
                        # swap r should be adjusted, c should be norm
                        self.E[[c+adjust,r]]=self.E[[r,c+adjust]]
                        done = True
                        #print('Swapping {},{}'.format(r,c+adjust))
                        break
                if not done:
                    # we did not find any row to switch..
                    # so know....we want to continue, but with the row adjusted 
                    # instead of [2,2], we look for [2,3] i.e. we go to the next
                    # column but adjust the row
                    adjust-=1
                    continue
            for r in range(self.l):
                # found a pivot
                if r==c+adjust:
                    dep.append(c)
                    continue
                else:
                    if self.E[r,c]==1:
                        self.E[r,:]+=self.E[c+adjust,:]
                        self.E = np.mod(self.E,2)
        if self.verbose:
            print('Parity check matrix in RRE form:')
            c = 0
            for i in range(self.E.shape[0]):
                if np.linalg.norm(self.E[i,:])>0:
                    print(self.E[i,:])
                else:
                    c+=1 
            print('...with {} trivial rows.'.format(c))

        # done with gaussian elimination! 

        tE = self.E[:2*self.N,:2*self.N]
        #null = la.null_space(tE)
        #dimN =null.shape[1]
        xs = sy.numbered_symbols('x')
        Xv = sy.zeros(2*self.N,1)
        dep_var = []
        ind_var = []
        for c in range(2*self.N):
            x= next(xs)
            Xv[c,0]=x
            if c in dep:
                dep_var.append(x)
            else:
                ind_var.append(x)
        #print(ind_var) # these are elements of null space
        dimN = len(ind_var)
        if self.verbose:
            print('Found {} vectors in the null space.'.format(dimN))
        vecs = np.zeros((2*self.N,dimN))
        nullB = Operator()
        for n in range(dimN):
            if self.verbose:
                print('Finding vector {}...'.format(n+1))
            new = tE*Xv
            xn = Xv
            for m,d in enumerate(ind_var):
                if m==n:
                    new = new.subs(d,1)
                    xn = xn.subs(d,1)
                else:
                    new = new.subs(d,0)
                    xn = xn.subs(d,0)
            sol = sy.solve(new)
            vecs[:,n]= np.mod(xn.subs(sol).T,2)
            #vecs[:,n]= apply_h(vecs[:,n])
            p = ''
            for s in range(self.N):
                c1 = vecs[s,n]
                c2 = vecs[s+self.N,n]
                if c1==0 and c2==0:
                    p+='I'
                elif c1==1 and c2==0:
                    p+='X'
                elif c1==0 and c2==1:
                    p+='Z' 
                elif c1==1 and c2==1:
                    p+='Y'
            if self.verbose:
                print('...tau is {}'.format(p))
            nullB+= PauliString(p,1)
        null = np.mod(vecs,2)
        dimE = 2*self.N-dimN
        self.dimE = dimE
        self.dimN = dimN
        self.null_vecs = null
        self.null_basis = nullB

    def get_transformation(self,qubits='default'):
        if qubits=='default':
            qubits = [i for i in range(self.dimN)]
        U = Operator()
        Ut = Operator()
        U+=PauliString('I'*self.N,1)
        Ut+=PauliString('I'*self.N,1)
        for i in range(self.dimN):
            I = self.dimN-i-1
            x = 'I'*qubits[i]+'X'+'I'*(self.N-qubits[i]-1)
            xt = 'I'*qubits[I]+'X'+'I'*(self.N-qubits[I]-1)
            UiT = Operator()

            Ui = Operator()
            use=True
            x = PauliString(x,1/np.sqrt(2))
            xt = PauliString(xt,1/np.sqrt(2))
            # check 
            for j in range(self.dimN):
                test= PauliString(self.null_basis.op[j].s,1)
                if j==i:
                    if x.comm(test):
                        print(test,x,'commuting')
                        use=False
                        break
                else:
                    if not x.comm(test):
                        print(test,x,'anti commuting')
                        use=False
                        break
            if not use:
                sys.exit('Wrong qubits specified for generators.')
            Ui+= PauliString(self.null_basis.op[i].s,1/np.sqrt(2))
            Ui+= x
            UiT+= PauliString(self.null_basis.op[I].s,1/np.sqrt(2))
            UiT+= xt

            U*=Ui
            Ut*=UiT
        return U,Ut

