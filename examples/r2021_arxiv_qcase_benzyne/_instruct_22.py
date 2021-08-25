import numpy as np
from hqca.core import *
from hqca.core.primitives import *
from hqca.tools import *
import sys
from numpy import sin as sin
from numpy import cos as cos
from copy import deepcopy as copy

class ExpPauli:
    def __init__(self,vec):
        v = np.asmatrix(vec)
        if v.shape[0]>v.shape[1]:
            v = v.T
        if np.linalg.norm(v)==0:
            self.iden=True
            self.a = 0
            self.v = v
        else:
            self.iden=False
            self.a = np.linalg.norm(v)
            self.v = v/self.a
    
    def __mul__(self,w):
        if self.iden:
            return w
        if w.iden:
            return self
        cc = np.cos(self.a)*np.cos(w.a)
        cs = np.cos(self.a)*np.sin(w.a)
        sc = np.sin(self.a)*np.cos(w.a)
        ss = np.sin(self.a)*np.sin(w.a)
        c = np.arccos(cc-np.dot(self.v,w.v.T)*ss)
        k1 = self.v*sc
        k2 = w.v*cs
        k3 = -np.cross(self.v,w.v)*ss
        k = (1/np.sin(c))*(k1+k2+k3)
        return ExpPauli(c*k)

    def __str__(self):
        t = '||v||: {:.5f}, '.format(self.a)
        t+= 'nx: {:+.5f}, '.format(self.v[0,0])
        t+= 'ny: {:+.5f}, '.format(self.v[0,1])
        t+= 'nz: {:+.5f}'.format(self.v[0,2])
        return t

    def matrix(self):
        x = np.matrix([[0,1],[1,0]],dtype=np.complex_)
        y = np.matrix([[0,-1j],[1j,0]],dtype=np.complex_)
        z = np.matrix([[1,0],[0,-1]],dtype=np.complex_)
        nx,ny,nz = self.v[0,0],self.v[0,1],self.v[0,2]
        i = np.identity(2)
        if self.iden:
            return np.identity(2)
        return np.cos(self.a)*i + (x*nx+y*ny+z*nz)*1j*np.sin(self.a)

    def U3(self):
        if self.iden:
            return 0,0,0
        A = np.sin(self.a)**2
        nx,ny,nz = self.v[0,0],self.v[0,1],self.v[0,2]
        part = nx**2+ny**2
        vd = np.cos(self.a)+1j*nz*np.sin(self.a)
        vo = (1j*nx-ny)*np.sin(self.a)
        if abs(part-0)<=1e-10:
            theta= 0
            sigma = (1j*np.log(vd)).real
            delta= 0
        else:
            theta = 2*np.arcsin(np.sqrt((nx**2+ny**2)*A))
            aleph=-ny*np.sin(self.a)/np.sin(theta/2)
            beta = nx*np.sin(self.a)/np.sin(theta/2)
            delta = (-1j*np.log(vo/np.sin(theta/2))).real
            sigma = (1j*np.log(vd/np.cos(theta/2))).real
        return theta,sigma+delta,sigma-delta

class BenzyneInstruct(Instructions):
    '''
    type 1, 2 and 3
    '''
    def __init__(self,operator,
            Nq,
            propagate=False,
            HamiltonianOperator=[],
            scaleH=1,
            **kw):
        if not Nq==1:
            sys.exit('Did not 1 qubit in instructions...')
        para = np.array([0.0,0.0,0.0])
        expS = ExpPauli(para)
        for A in operator:
            para = np.array([0.0,0.0,0.0])
            for o in A:
                if o.s=='X':
                    para[0]=np.imag(o.c)
                elif o.s=='Y':
                    para[1]=np.imag(o.c)
                elif o.s=='Z':
                    para[2]=np.imag(o.c)
            expS = ExpPauli(para)*expS
        #
        paraH = np.array([0.0,0.0,0.0])
        for o in HamiltonianOperator:
            if o.s=='X':
                paraH[0]= np.real(o.c)*scaleH
            elif o.s=='Y':
                paraH[1]=np.real(o.c)*scaleH
            elif o.s=='Z':
                paraH[2]=np.real(o.c)*scaleH
        expiH = ExpPauli(paraH)
        exp = expiH*expS
        self._gates = [
                [(exp,),self._U3]
                ]

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

    def _U3(self,Q,exp):
        theta,phi,lamb = exp.U3()
        Q.U3(0,theta,phi,lamb)

