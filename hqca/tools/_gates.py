import numpy as np
from numpy import kron as k
from functools import reduce
import sympy as sy
np.set_printoptions(suppress=True,precision=4,linewidth=200)
#import pyzx as pz


class Circ:
    def __init__(self,size=3,symbolic=False):
        if symbolic:
            pass
            from sympy import cos,sin
        elif not symbolic:
            from numpy import cos,sin
        self.cos = cos
        self.sin = sin
        self.n = size
        self.N = 2**self.n
        self.b = ['{:0{}b}'.format(i,self.n)[::1] for i in range(0,self.N)]
        self.m = np.asarray(np.identity(self.N))
        self.num_sqg = 0
        self.num_cx  = 0 
        self._cN_basis()
        self.c = None
        self.qasm = '\n\n # circuit qasm \n'
        self.qasm+= 'qubits {}\n\n'.format(self.n)


    def trace_operator(self,qb=[0]): 
        '''
        trace over the qubits  in qb

        should be listed in reverse order
        '''
        nd = self.n-len(qb)
        Nd = len(qb)
        nb = ['{:0{}b}'.format(i,nd)[::1] for i in range(0,2**nd)]
        nbd= {'{:0{}b}'.format(i,nd)[::1]:i for i in range(0,2**nd)}
        keys= {'{:0{}b}'.format(i,self.n)[::1]:i for i in range(0,2**self.n)}
        Nb = ['{:0{}b}'.format(i,Nd)[::1] for i in range(0,2**Nd)]
        new = np.zeros((2**nd,2**nd))
        for n,i in enumerate(nb): #  new
            for m,j in enumerate(nb):
                for basis in Nb:
                    ket = '0'*self.n
                    bra = '0'*self.n
                    q,s = 0,0
                    for r in range(self.n):
                        if not r in qb:
                            t = i[s]
                            u = j[s]
                            s+=1
                        else:
                            t = basis[q]
                            u = basis[q]
                            q+=1
                        bra = bra[:r]+t+bra[r+1:]
                        ket = ket[:r]+u+ket[r+1:]
                    ind_ket = keys[ket]
                    ind_bra = keys[bra]
                    new[n,m]+= self.m[ind_ket,ind_bra]
        circ = Circ(nd)
        circ.m = new*(1/2**(len(qb)))
        return circ


    def _cN_basis(self):
        self.cNb = []
        self.cNi = []
        for i in range(0,self.n+1):
            for n,b in enumerate(self.b):
                temp = 0
                for s in b:
                    temp+=int(s)
                if temp==i:
                    self.cNb.append(b)
                    self.cNi.append(n)

    def get_cN(self):
        t1 = np.zeros(self.m.shape,dtype=np.complex_)
        t2 = np.zeros(self.m.shape,dtype=np.complex_)
        for n,r in enumerate(self.cNi):
            t1[n,:]=self.m[r,:]
        for n,c in enumerate(self.cNi):
            t2[:,n]=t1[:,c]
        s = '   '
        for i in self.cNb:
            s+= i +'   '
        return t2


    def get(self):
        return self.m

    def __str__(self):
        print(np.real(self.m))
        if np.count_nonzero(np.imag(self.m))>0:
            print(np.imag(self.m))
        return ''

    def _apply_sqg(self,g,i):
        temp = np.identity(1)
        for j in range(0,i):
            temp = np.kron(temp,np.identity(2))
        temp = np.kron(temp,g)
        for j in range(i+1,self.n):
            temp = np.kron(temp,np.identity(2))
        self.m = reduce(np.dot, (temp,self.m))
        self.num_sqg+=1 



    def gate(self,gate=None,*args,**kwargs):
        if type(gate)==type(None):
            pass
        elif gate=='i':
            pass
        elif gate=='cx':
            self.Cx(*args,**kwargs)
        elif gate=='cz':
            self.Cz(*args,**kwargs)
        elif gate=='sw':
            self.Sw(*args,**kwargs)
        elif gate=='h':
            self.h(*args,**kwargs)
        elif gate=='s':
            self.s(*args,**kwargs)
        elif gate=='si':
            self.si(*args,**kwargs)
        elif gate=='x':
            self.x(*args,**kwargs)
        elif gate=='y':
            self.y(*args,**kwargs)
        elif gate=='z':
            self.z(*args,**kwargs)


    def Ch(self,i,j,nCx=2):
        '''
        Controlled Hadamard Gate: note there are actually two implementations: a
        single Cx gate and double Cx gate implementation. 

        Single is as (note p=pi):
        Ry(j,-p/4)Cx(i,j)Ry(j,p/4)|...i...j...>

        Where as the two CNOT sequence is:
        Ph(i,p/2)Rz(j,-p/2)Cx(i,j)Rz(j,-p/2)Ry(j,-p/4)Cx(i,j)Ry(j,p/4)Rz(j,p)|>

        Note we have....6 sqg  and 2 CNOT versus 1 CNOT and 2 sqg.
        '''
        if nCx==2:
            self.Rz(j,np.pi)
            self.Ry(j,np.pi/4)
            self.Cx(i,j)
            self.Ry(j,-np.pi/4)
            self.Rz(j,-np.pi/2)
            self.Cx(i,j)
            self.Rz(j,-np.pi/2)
            self.ph(i,np.pi/2)
        elif nCx==1:
            self.Ry(j,np.pi/4)
            self.Cx(i,j)
            self.Ry(j,-np.pi/4)


    def Sw(self,i,j):
        '''
        Swap gate between two qubits i,j
        '''
        self.Cx(i,j)
        self.Cx(j,i)
        self.Cx(i,j)
        self.num_cx-=3

    def i(self,**kw):
        pass

    def SRy(self,i,j,theta):
        self.Cx(j,i)
        self.x(j)
        self.CRy(i,j,theta)
        self.x(j)
        self.Cx(j,i)

    def Sh(self,i,j,**kw):
        self.Cx(j,i)
        self.x(j)
        self.Ch(i,j,**kw)
        self.x(j)
        self.Cx(j,i)




    def Toff4(self,i,j,k,l):
        hold = np.identity(self.N)
        for x in range(0,self.N):
            if self.b[x][i]=='1' and self.b[x][j]=='1' and self.b[x][k]=='1':
                for y in range(0,self.N):
                    c1 = (self.b[x][0:l]==self.b[y][0:l])
                    c2 = (self.b[x][l+1:]==self.b[y][l+1:])
                    if c1 and c2:
                        if self.b[x][l]=='0' and self.b[y][l]=='0':
                            hold[x,y]=0
                        elif self.b[x][l]=='0' and self.b[y][l]=='1':
                            hold[x,y]=1
                        elif self.b[x][l]=='1' and self.b[y][l]=='0':
                            hold[x,y]=1
                        elif self.b[x][l]=='1' and self.b[y][l]=='1':
                            hold[x,y]=0
                        else:
                            pass
                        continue
        self.m = reduce(np.dot, (hold,self.m))
        self.num_cx+=36
        self.num_sqg+=81

    def CCS(self,i,j,k,l):
        self.Toff4(i,j,l,k)
        self.Toff4(i,j,k,l)
        self.Toff4(i,j,l,k)


    def Toff(self,i,j,k):
        hold = np.identity(self.N)
        for x in range(0,self.N):
            if self.b[x][i]=='1' and self.b[x][j]=='1':
                for y in range(0,self.N):
                    c1 = (self.b[x][0:k]==self.b[y][0:k])
                    c2 = (self.b[x][k+1:]==self.b[y][k+1:])
                    if c1 and c2:
                        if self.b[x][k]=='0' and self.b[y][k]=='0':
                            hold[x,y]=0
                        elif self.b[x][k]=='0' and self.b[y][k]=='1':
                            hold[x,y]=1
                        elif self.b[x][k]=='1' and self.b[y][k]=='0':
                            hold[x,y]=1
                        elif self.b[x][k]=='1' and self.b[y][k]=='1':
                            hold[x,y]=0
                        else:
                            pass
                        continue
        self.m = reduce(np.dot, (hold,self.m))
        self.num_cx+=6
        self.num_sqg+=9

    def Fred(self,i,j,k):
        self.Toff(i,k,j)
        self.Toff(i,j,k)
        self.Toff(i,k,j)

    def h(self,i,**kw):
        a = 1/np.sqrt(2)
        h = np.array([[a,a],[a,-a]])
        self._apply_sqg(h,i)

    def sx(self,i,**kw):
        a = 1/2
        sx = a*np.array([[1+1j, 1-1j],[1-1j,1+1j]])
        self._apply_sqg(sx,i)

        # square root x

    def z(self,i,**kw):
        z = np.array([[1,0],[0,-1]])
        self._apply_sqg(z,i)

    def x(self,i,**kw):
        x = np.array([[0,1],[1,0]])
        self._apply_sqg(x,i)

    def y(self,i,**kw):
        y = np.array([[0,-1j],[1j,0]])
        self._apply_sqg(y,i)

    def ph(self,i,theta=np.pi/2,**kw):
        c = self.cos(theta)
        s = 1j*self.sin(theta)
        ph = np.array([[1,0],[0,c+s]])
        self._apply_sqg(ph,i)

    def particle(self,i,**kw):
        mat = np.array([[0,0],[0,1]])
        self._apply_sqg(mat,i)

    def hole(self,i,**kw):
        mat = np.array([[1,0],[0,0]])
        self._apply_sqg(mat,i)

    def create(self,i,**kw):
        mat = np.array([[0,0],[1,0]])
        self._apply_sqg(mat,i)

    def annihilate(self,i,**kw):
        mat = np.array([[0,1],[0,0]])
        self._apply_sqg(mat,i)

    def s(self,i,**kw):
        self.ph(i,theta=np.pi/2)

    def si(self,i,**kw):
        self.ph(i,theta=-np.pi/2)

    def t(self,i,**kw):
        self.ph(i,theta=np.pi/4)

    def ti(self,i,**kw):
        self.ph(i,theta=-np.pi/4)

    def t2(self,i,**kw):
        self.ph(i,theta=np.pi/8)

    def t2i(self,i,**kw):
        self.ph(i,theta=-np.pi/8)

    def phc(self,i,theta=np.pi/2,**kw):
        c = self.cos(theta)
        s = 1j*self.sin(theta)
        ph = np.array([[c+s,0],[0,c+s]])
        self._apply_sqg(ph,i)


    def Cy(self,i,j,**kw):
        self.si(j)
        self.Cx(i,j)
        self.s(j)

    def Cx(self,i,j,**kw):
        hold = np.identity(self.N)
        for k in range(0,self.N):
            if self.b[k][i]=='1':
                for l in range(0,self.N):
                    c1 = (self.b[k][0:j]==self.b[l][0:j])
                    c2 = (self.b[k][j+1:]==self.b[l][j+1:])
                    if c1 and c2:
                        if self.b[k][j]=='0' and self.b[l][j]=='0':
                            hold[k,l]=0
                        elif self.b[k][j]=='0' and self.b[l][j]=='1':
                            hold[k,l]=1
                        elif self.b[k][j]=='1' and self.b[l][j]=='0':
                            hold[k,l]=1
                        elif self.b[k][j]=='1' and self.b[l][j]=='1':
                            hold[k,l]=0
                        else:
                            pass
                        continue
        self.m = reduce(np.dot, (hold,self.m))
        self.num_cx+=1 

    def Cxr(self,i,j):
        self.x(i)
        self.Cx(i,j)
        self.x(i)

    def Cs(self,i,j):
        self.Cx(i,j)
        self.Rz(j,-np.pi/4)
        self.Cx(i,j)
        self.Rz(j,np.pi/4)
        self.ph(i,np.pi/4)

    def Cz(self,i,j,**kw):
        self.h(j)
        self.Cx(i,j)
        self.h(j)

    def CRy(self,i,j,theta):
        self.Ry(j,theta/2)
        self.Cx(i,j)
        self.Ry(j,-theta/2)
        self.Cx(i,j)


    def Rx(self,i=0,theta=np.pi/2,**kw):
        c = self.cos(theta/2)
        s = -1j*self.sin(theta/2)
        r = np.array([[c,s],[s,c]])
        self._apply_sqg(r,i)

    def Rz(self,i=0,theta=np.pi/2,**kw):
        c = self.cos(theta/2)
        s = 1j*self.sin(theta/2)
        r = np.array([[c-s,0],[0,c+s]])
        self._apply_sqg(r,i)

    def Ry(self,i=0,theta=np.pi/2,**kw):
        c = self.cos(theta/2)
        s = self.sin(theta/2)
        r = np.array([[c,-s],[s,c]])
        self._apply_sqg(r,i)



