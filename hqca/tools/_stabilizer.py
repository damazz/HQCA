import numpy as np
from scipy import linalg as la
np.set_printoptions(linewidth=300)
import sympy as sy
from hqca.operators import *
from hqca.core.primitives import *

class Stabilizer:
    def __init__(self,paulis,verbose=True,**kw):
        '''
        Given a Pauli operator (i.e., Operator class composed of Pauli strings),
        we can represent the check sum representation of the Pauli matrix, and then
        attempt to find and potential symmetry generators. 

        In particular, run:
        zed = Stabilizer(paulis)
        zed.gaussian_elimination()
        zed.find_symmetry_generators()

        '''
        self.verbose = verbose
        self.op = paulis
        if type(paulis)==type([]):
            self.N = len(paulis[0])
            self.l = len(paulis)
            self.G = np.zeros((2*self.N,self.l))
            iden = 'I'*self.N
            add = 0
            for n,i in enumerate(paulis):
                if i==iden:
                    add = -1
                    continue
                else:
                    self.G[:,n+add]=self._pauli_to_check(i)
        else:
            self.N = len(next(iter(paulis)).s)
            self.l = len(paulis)
            self.G = np.zeros((2*self.N,self.l))
            iden = 'I'*self.N
            add = 0
            for n,i in enumerate(paulis.keys()):
                if i==iden:
                    add = -1
                    continue
                else:
                    self.G[:,n+add]=self._pauli_to_check(i)
        if add==-1:
            self.G = self.G[:,:self.l-1]
            self.l-=1
        self.G0 = np.copy(self.G)
        self.E = np.zeros((self.l,2*self.N))
        self.E[:,:self.N] = self.G[self.N:,:].T
        self.E[:,self.N:] = self.G[:self.N,:].T
        if self.verbose:
            print('Parity check matrix: ')
            print(self.E)
            print('Generator matrix: ')
            print(self.G)

    def _pauli_to_check(self,p):
        vec = np.zeros(2*self.N)
        for i in range(self.N):
            if p[i]=='Z' or p[i]=='Y':
                vec[i]=1
        for i in range(self.N,2*self.N):
            if p[i%self.N]=='X' or p[i%self.N]=='Y':
                vec[i]=1
        return vec

    def _check_to_pauli(self,vec):
        s = ''
        for i in range(self.N):
            if vec[i]==1:
                if vec[i+self.N]==1:
                    s+='Y'
                else:
                    s+='Z'
            else:
                if vec[i+self.N]==1:
                    s+='X'
                else:
                    s+='I'
        return PauliString(s,1)

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
        pivot = []
        for c in range(2*self.N):
            if (c+adjust)==self.l:
                self.done=True
                break
            if self.E[c+adjust,c]==1:
                pass
            else:
                done = False
                for r in range(c+1+adjust,self.l):
                    # for rows in the column range
                    #
                    #
                    if self.E[r,c]==1:
                        #
                        # swap r should be adjusted, c should be norm
                        #
                        self.E[[c+adjust,r]]=self.E[[r,c+adjust]]
                        done = True
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
                    pivot.append(c)
                    # dependent is simply the amount of dependent variables 
                    continue
                else:
                    if self.E[r,c]==1:
                        self.E[r,:]+=self.E[c+adjust,:]
                        self.E = np.mod(self.E,2)
        if self.verbose:
            print('Parity check matrix in RRE form:')
        c = 0
        n = 0
        for i in range(self.E.shape[0]):
            if np.linalg.norm(self.E[i,:])>0:
                if self.verbose:
                    print(self.E[i,:])
                n+=1 
            else:
                c+=1 
        if self.verbose:
            print('...with {} trivial rows.'.format(c))
        self.pivot = pivot
        self.Gm = np.zeros((2*self.N,n)) #G-mod
        self.Gm[:self.N,:] = self.E[:n,self.N:].T
        self.Gm[self.N:,:] = self.E[:n,:self.N].T
        self.paulis = {}
        for v in range(self.G0.shape[1]):
            target = self.G0[:,v]
            done = False
            soln=  []
            while not done:
                done = True
                for u in range(self.Gm.shape[0]):
                    if target[u]==0:
                        continue
                    for w in range(self.Gm.shape[1]):
                        # target is 1
                        if self.Gm[u,w]==1:
                            target = np.mod(target+self.Gm[:,w],2)
                            soln.append(w)
                            done=False
                            break
                        # checking for 1s 
            #for u in range(self.Gm.shape)
            p0 = self._check_to_pauli(self.G0[:,v])
            ps = [self._check_to_pauli(self.Gm[:,i]) for i in soln]
            c = 1
            prod = PauliString('I'*self.N,1)
            for i in ps:
                prod = prod*i
            self.paulis[p0.s] = [[s.s for s in ps],prod.c]

        # done with gaussian elimination! 
        # 
        # now, we try to find symmetry generators
        # for the space of operators
        #
        # i.e., a basis set for the null space

    def find_symmetry_generators(self):
        tE = self.E[:2*self.N,:2*self.N] ## tE is a square matrix? yes.
        # 
        #dimension should be square....I think
        # if there are 10 non trivial rows, 6 qubits, we should have 2 non
        # trivial generators 
        #
        #

        xs = sy.numbered_symbols('x')
        zs = sy.numbered_symbols('z')
        X = sy.zeros(2*self.N,1)
        pivot_var = []
        indep_var = []
        variables = []
        for c in range(self.N):
            x= next(zs)
            if c in self.pivot:
                pivot_var.append(x)
            else:
                indep_var.append(x)
            variables.append(x)
        for c in range(self.N,self.N*2):
            x = next(xs)
            if c in self.pivot:
                pivot_var.append(x)
            else:
                indep_var.append(x)
            variables.append(x)
        #print(pivot_var,indep_var)
        dimN = self.N*2-len(pivot_var)
        if self.verbose:
            print('Rank of null space: {}'.format(dimN))
        X = sy.zeros(2*self.N,1) #here is our x vector
        Np = 0 
        for n in range(2*self.N):
            if n in self.pivot:
                for m in range(n+1,2*self.N):
                    if m in self.pivot:
                        pass
                    else:
                        X[n]+= self.E[Np,m]*variables[m]
                Np+=1 
            else:
                X[n]=variables[n]
        # now.....we can separate N according to dep variables? 
        # Xv will hold the new variables 
        Xv = sy.zeros(2*self.N,dimN)
        for i in range(dimN):
            temp = X[:,0]
            for j in range(dimN):
                if i==j:
                    temp = temp.subs(indep_var[j],int(1))
                else:
                    temp = temp.subs(indep_var[j],0)
            Xv[:,i]=temp[:,0]

        #sy.pprint(Xv)
        #new = tE
        # now, need to check that they are all linear
        # and that they commute with all terms in H

        Xv = np.asarray(Xv)
        def dot(a,b):
            bp = np.zeros(b.shape)
            bp[:self.N]=b[self.N:]
            bp[self.N:]=b[:self.N]
            return np.mod(np.dot(a.T,bp),2)


        #new = np.zeros(Xv.shape)
        #OBfor n in range(0,dimN):
        #    temp = np.copy(Xv[:,n])
        #    #print('-----')
        #    #print(temp)
        #    for j in range(n):
        #        #print('temp: ')
        #        #print(temp)
        #        #temp2 = np.zeros(Xv[:,j].shape)
        #        t = np.copy(new[:,j])
        #        #temp2[:self.N] = Xv[self.N:,j]
        #        #temp2[self.N:] = Xv[:self.N,j]
        #        temp+= np.mod(dot(Xv[:,n],t)*t,2)
        #        #print( dot(Xv[:,j],Xv[:,n]))
        #        #print(Xv[:,n],Xv[:,j])
        #    new[:,n]=np.mod(temp[:],2)
        #for n in range(dimN):
        #    for m in range(dimN):
        #        print(n,m,dot(new[:,n],new[:,m]))

        #print(new)
        #
        #Xv = np.copy(new)
        nullB = []
        ## assuming that they work....
        for n in range(1,dimN):
            pass
        for n in range(dimN):
            if self.verbose:
                print('Finding vector {}...'.format(n+1))
            p = ''
            for s in range(self.N):
                c1 = Xv[s,n]
                c2 = Xv[s+self.N,n]
                if c1==0 and c2==0:
                    p+='I'
                elif c1==1 and c2==0:
                    p+='Z'
                elif c1==0 and c2==1:
                    p+='X' 
                elif c1==1 and c2==1:
                    p+='Y'
            if self.verbose:
                print('...tau is {}'.format(p))
            nullB.append(p)
        dimE = 2*self.N-dimN
        self.dimE = dimE
        self.dimN = dimN
        self.null_vecs = Xv
        self.null_basis = nullB

class StabilizedCircuit(Stabilizer):
    def construct_circuit(self):
        try: 
            self.null_basis
        except Exception as e:
            sys.exit('Need to run symmetry generation first.')
        z_symm = []
        z_str = []
        num_symm = []
        for i in range(self.dimN):
            nz = np.count_nonzero(self.null_vecs[self.N:,i])
            if nz==0:
                Nz = np.count_nonzero(self.null_vecs[:self.N,i])
                if Nz>1:
                    z_symm.append(i)
                    z_str.append(self._check_to_pauli(self.null_vecs[:,i]))
        new = np.zeros((self.dimE+len(z_symm),2*self.N))
        new[:self.dimE,:]=self.Gm[:,:].T
        for i in range(self.dimE):
            pass
        for n,i in enumerate(z_symm):
            new[self.dimE+n,:]=self.null_vecs[:,i].T
        self.m = new
        self.T0 = [] # T is transformed measurement, M is native (Z)
        for i in range(new.shape[0]):
            self.T0.append(self._check_to_pauli(new[i,:]))
        self.T_M = {}
        self.gates = []
        self.zz = z_str


    def simplify(self):
        m = self._simplify_y_qwc(self.m)
        m = self._simplify_xx_zz_qwc(m)
        m = self._simplify_x_z_qwc(m)
        self.m = m
        self.T_M = {self.T0[i].s:self._check_to_pauli(self.m[i,:]).s
                for i in range(self.m.shape[0])}
        #print('T -> M')
        #print(self.T_M)

    def _simplify_y_qwc(self,mat,verbose=True):
        '''
        note, this only works for QWC types..otherwise more complicated
        procedure required to get rid of y-type ops
        '''
        for i in range(self.N):
            for j in range(mat.shape[0]):
                if mat[j,i]==1 and mat[j,i+self.N]==1:
                    self.gates.append([
                                (i,),
                                apply_si])
                    for k in range(mat.shape[0]):
                        if mat[k,i]==1 and mat[k,i+self.N]==1:
                            mat[k,i]=0
                        elif mat[k,i]==0 and mat[k,i+self.N]==1:
                            mat[k,i]=1
                    break
        return mat

    def _simplify_xx_zz_qwc(self,mat,verbose=True):
        # find rank of each qubit 
        # basically, we want to go from high rank to low rank
        done = False
        def _sub(mat):
            rank = np.zeros(2*self.N)
            for i in range(2*self.N):
                rank[i]=np.sum(mat[:,i])
            for r in range(mat.shape[0]):
                for i in range(self.N):
                    for j in range(i):
                        # check for zz, xx
                        # mostly, xx
                        #c1 = mat[r,i]==1 and mat[r,j]==1
                        c2 = mat[r,i+self.N]==1 and mat[r,j+self.N]==1
                        if c2:
                            l1,l2 = rank[i],rank[j]
                            r1,r2 = rank[i+self.N],rank[j+self.N]
                            if l1>l2 or r2>r1:
                                #self.gates.append(['Cx',[i,j]])
                                self.gates.append([
                                            (i,j),
                                            apply_cx])
                                for s in range(mat.shape[0]):
                                    mat[s,i]+=mat[s,j]
                                    mat[s,j+self.N]+=mat[s,i+self.N]
                            else:
                                #self.gates.append(['Cx',[j,i]])
                                self.gates.append([
                                            (j,i),
                                            apply_cx])
                                for s in range(mat.shape[0]):
                                    mat[s,j]+=mat[s,i]
                                    mat[s,i+self.N]+=mat[s,j+self.N]
                            return np.mod(mat,2),False
            return mat,True
        iters = 0
        while (not done) or iters<5:
            mat,done = _sub(mat)
            iters+=1 
        return mat
            
    def _simplify_x_z_qwc(self,mat):
        '''
        note, this only works for QWC types..otherwise more complicated
        procedure required to get rid of y-type ops
        '''
        for i in range(self.N):
            for j in range(mat.shape[0]):
                if mat[j,i]==0 and mat[j,i+self.N]==1:
                    self.gates.append(
                            [
                                (i,),
                                apply_h])
                    for k in range(mat.shape[0]):
                        if mat[k,i]==0 and mat[k,i+self.N]==1:
                            mat[k,i]=1
                            mat[k,i+self.N]=0
                        elif mat[k,i]==1 and mat[k,i+self.N]==0:
                            mat[k,i]=0
                            mat[k,i+self.N]=1
                    break
        return mat


