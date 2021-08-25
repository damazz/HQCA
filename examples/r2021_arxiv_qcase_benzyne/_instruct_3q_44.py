import numpy as np
from hqca.core import *
from hqca.core.primitives import *
from hqca.tools import *
import sys
from math import pi
from collections import OrderedDict

class Line3Q(Instructions):
    '''
    type 1, 2 and 3
    '''
    def __init__(self,operator,
            Nq,propagate=False,**kw):
        self.use = []
        self._gates = []
        mapping = OrderedDict()
        mapping['d1']=set(['IYX','IXY'])
        mapping['d2']=set(['YXI','XYI'])
        mapping['a1']=set(['XIY','ZYZ'])
        mapping['a2']=set(['IYX','YZZ'])
        mapping['a3']=set(['XYI','ZZY'])
        mapping['b1']=set(['YIX','ZYZ'])
        mapping['b2']=set(['IXY','YZZ'])
        mapping['b3']=set(['YXI','ZZY'])
        mapping['e1']=set(['YII','YZZ'])
        mapping['e2']=set(['IYI','ZYZ'])
        mapping['e3']=set(['IIY','ZZY'])
        mapping['g2']=set(['IIY'])
        mapping['g4']=set(['YII'])
        mapping['g6']=set(['IYI'])
        mapping['f1']=set(['IYX'])
        mapping['f2']=set(['IXY'])
        mapping['f5']=set(['YXI'])
        mapping['f6']=set(['XYI'])
        mapping['g1']=set(['ZZY'])
        mapping['g3']=set(['YZZ'])
        mapping['g5']=set(['ZYZ'])
        mapping['d3']=set(['YIX','XIY'])
        mapping['f3']=set(['YIX'])
        mapping['f4']=set(['XIY'])
        self.mapping = mapping
        # iterating over each macro anastz A in the operator S
        for A in operator:
            # for each A we will remove elements until we account 
            # for the entire ansatz
            #
            use = []
            keys = set(A.keys())
            while len(keys)>0:
                for k,v in mapping.items():
                    if v.issubset(keys):
                        use.append(k)
                        for i in v: #remove elements
                            keys.remove(i)
            get_funcs = {
                    'a1':self._type_A1,
                    'a2':self._type_A2,
                    'a3':self._type_A3,
                    'b1':self._type_B1,
                    'b2':self._type_B2,
                    'b3':self._type_B3,
                    'd1':self._type_D1,
                    'd2':self._type_D2,
                    'd3':self._type_D3,
                    'e1':self._type_E1,
                    'e2':self._type_E2,
                    'e3':self._type_E3,
                    'f1':self._type_F1,
                    'f2':self._type_F2,
                    'f3':self._type_F3,
                    'f4':self._type_F4,
                    'f5':self._type_F5,
                    'f6':self._type_F6,
                    'g1':self._type_G1,
                    'g2':self._type_G2,
                    'g3':self._type_G3,
                    'g4':self._type_G4,
                    'g5':self._type_G5,
                    'g6':self._type_G6,
                    }
            for u in use:
                self._gates.append(
                        [(A,),get_funcs[u]]
                        )
            self.use+= use
        print(self.use)
        if propagate:
            self._applyH(Nq=Nq,**kw)

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

    def _type_A1(self,Q,op):
        paulis = ['XIY','ZYZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.Rz(2,-pi/2)
        Q.sx(1)
        Q.s(1)
        Q.s(0)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.sx(0)
        Q.Rz(2,para[1])
        Q.Rz(0,para[0]+pi)
        Q.sx(0)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.s(1)
        Q.sx(1)
        Q.z(1)
        Q.s(2)
        Q.s(0)

    def _type_A2(self,Q,op):
        paulis = ['IYX','YZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.si(1)
        Q.sx(0)
        Q.s(2)
        Q.Cx(2,1)
        Q.Cx(0,1)
        Q.sx(2)
        Q.Rz(1,para[1])
        Q.Rz(2,para[0]+pi)
        Q.sx(2)
        Q.Cx(0,1)
        Q.Cx(2,1)
        Q.z(0)
        Q.sx(0)
        Q.z(0)
        Q.s(1)
        Q.s(2)

    def _type_A3(self,Q,op):
        paulis = ['XYI','ZZY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.si(1)
        Q.sx(2)
        Q.s(0)
        Q.Cx(2,1)
        Q.Cx(0,1)
        Q.sx(0)
        Q.Rz(1,para[1])
        Q.Rz(0,para[0]+pi)
        Q.sx(0)
        Q.Cx(0,1)
        Q.Cx(2,1)
        Q.z(2)
        Q.sx(2)
        Q.z(2)
        Q.s(1)
        Q.s(0)

    def _type_B1(self,Q,op):
        paulis = ['YIX','ZYZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(1)
        Q.s(1)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.sx(0)
        Q.Rz(2,para[1])
        Q.Rz(0,para[0]+pi)
        Q.sx(0)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.s(1)
        Q.sx(1)
        Q.z(1)
        Q.z(0)

    def _type_B2(self,Q,op):
        paulis = ['IXY','YZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(0)
        Q.z(0)
        Q.Cx(2,1)
        Q.Cx(0,1)
        Q.sx(2)
        Q.Rz(1,para[1])
        Q.Rz(2,para[0]+pi)
        Q.sx(2)
        Q.Cx(0,1)
        Q.Cx(2,1)
        Q.sx(0)
        Q.z(0)
        Q.z(2)

    def _type_B3(self,Q,op):
        paulis = ['YXI','ZZY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(2)
        Q.Cx(2,1)
        Q.Cx(0,1)
        Q.sx(0)
        Q.Rz(1,para[1])
        Q.Rz(0,para[0]+pi)
        Q.sx(0)
        Q.Cx(0,1)
        Q.Cx(2,1)
        Q.z(2)
        Q.sx(2)
        Q.z(2)
        Q.z(0)

    def _type_F1(self,Q,op):
        paulis = ['IYX']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(1)
        Q.s(2)
        Q.sx(2)
        Q.s(2)
        Q.Cx(1,2)
        Q.Rz(2,para[0])
        Q.Cx(1,2)
        Q.z(1)
        Q.sx(1)
        Q.z(1)
        Q.s(2)
        Q.sx(2)
        Q.s(2)

    def _type_F2(self,Q,op):
        paulis = ['IXY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(1)
        Q.sx(1)
        Q.z(1)

        Q.sx(2)
        Q.s(2)
        Q.Cx(1,2)
        Q.Rz(2,para[0])
        Q.Cx(1,2)
        Q.sx(1)
        Q.s(1)
        Q.s(2)
        Q.sx(2)
        Q.z(2)

    def _type_F3(self,Q,op):
        paulis = ['YIX']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(0)
        Q.z(0)
        Q.s(2)
        Q.sx(2)
        Q.s(2)
        Q.Cx(0,2)
        Q.Rz(2,para[0])
        Q.Cx(0,2)
        Q.sx(0)
        Q.z(0)
        Q.s(2)
        Q.sx(2)
        Q.s(2)

    def _type_F4(self,Q,op):
        paulis = ['XIY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(0)
        Q.sx(0)

        Q.sx(2)
        Q.s(2)

        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rz(2,para[0])
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.Cx(1,2)

        Q.z(0)
        Q.sx(0)
        Q.s(0)

        Q.s(2)
        Q.sx(2)
        Q.z(2)
    
    def _type_F5(self,Q,op):
        paulis = ['YXI']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(1)
        Q.sx(1)
        Q.s(1)
        Q.sx(0)
        Q.z(0)
        Q.Cx(0,1)
        Q.Rz(1,para[0])
        Q.Cx(0,1)
        Q.s(1)
        Q.sx(1)
        Q.s(1)
        Q.sx(0)
        Q.z(0)

    def _type_F6(self,Q,op):
        paulis = ['XYI']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(0)
        Q.sx(0)
        Q.sx(1)
        Q.s(1)

        Q.Cx(0,1)
        Q.Rz(1,para[0])
        Q.Cx(0,1)

        Q.z(0)
        Q.sx(0)
        Q.s(0)

        Q.s(1)
        Q.sx(1)
        Q.z(1)

    def _type_G1(self,Q,op):
        paulis = ['ZZY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(2)
        Q.z(2)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rz(2,para[0])
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.sx(2)
        Q.z(2)


    def _type_G3(self,Q,op):
        paulis = ['YZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(0)
        Q.z(0)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rz(2,para[0])
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.sx(0)
        Q.z(0)

    def _type_G5(self,Q,op):
        paulis = ['ZYZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(1)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rz(2,para[0])
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.z(1)
        Q.sx(1)
        Q.z(1)

    def _type_G2(self,Q,op):
        paulis = ['IIY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(2)
        Q.Rz(2,para[0]+pi)
        Q.sx(2)
        Q.z(2)

    def _type_G6(self,Q,op):
        paulis = ['IYI']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(1)
        Q.Rz(1,para[0]+pi)
        Q.sx(1)
        Q.z(1)

    def _type_G4(self,Q,op):
        paulis = ['YII']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(0)
        Q.Rz(0,para[0]+pi)
        Q.sx(0)
        Q.z(0)

    def _type_D1(self,Q,op):
        '''
        '''
        paulis = ['IYX','IXY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(2)
        Q.sx(2)
        Q.sx(1)
        Q.Cx(2,1)
        Q.Rz(1, para[0]+pi)
        Q.sx(1)
        Q.z(1)
        
        Q.s(2)
        Q.sx(2)
        Q.Rz(2,-para[1]+pi)
        Q.sx(2)
        Q.si(2)
        
        Q.Cx(2,1)
        Q.sx(2)
        Q.s(2)

    def _type_D2(self,Q,op):
        '''
        '''
        paulis = ['YXI','XYI']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(1)
        Q.sx(1)
        Q.sx(0)
        Q.Cx(1,0)
        Q.Rz(0, para[0]+pi)
        Q.sx(0)
        Q.z(0)
        
        Q.s(1)
        Q.sx(1)
        Q.Rz(1,-para[1]+pi)
        Q.sx(1)
        Q.si(1)
        
        Q.Cx(1,0)
        Q.sx(1)
        Q.s(1)


    def _type_D3(self,Q,op):
        '''
        '''
        paulis = ['YIX','XIY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(2)
        Q.sx(2)
        Q.s(2)
        Q.si(2)
        Q.Cx(2,1)
        Q.Cx(1,0)
        Q.Cx(2,1)
        Q.Cx(1,0)
        Q.sx(0)
        Q.s(0)
        Q.Rz(0, para[0])
        Q.s(0)
        Q.sx(0)
        Q.s(0)
        Q.s(0)
        Q.s(2)
        Q.sx(2)
        Q.s(2)
        Q.Rz(2,-para[1])
        Q.s(2)
        Q.sx(2)
        Q.z(2)
        Q.Cx(1,0)
        Q.Cx(2,1)
        Q.Cx(1,0)
        Q.Cx(2,1)
        Q.z(2)
        Q.sx(2)
        Q.s(2)

    def _type_E1(self,Q,op):
        '''
        '''
        paulis = ['YII','YZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(0)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rz(0,para[0]) #YII
        Q.Rz(2,para[1]) #YZZ
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.sx(0)
        Q.x(0)

    def _type_E2(self,Q,op):
        '''
        '''
        paulis = ['IYI','ZYZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(1)
        Q.Rz(1,para[0]) #IYI
        Q.Rz(1,pi/2) #s
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rz(2,para[1]) #ZYZ
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.Rz(1,pi/2)
        Q.sx(1)
        Q.Rz(1,pi)

    def _type_E3(self,Q,op):
        '''
        '''
        paulis = ['IIY','ZZY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.sx(2)
        Q.Cx(2,1)
        Q.Cx(1,0)
        Q.Rz(2,para[0]) #IIY
        Q.Rz(0,para[1]) #ZZY
        Q.Cx(1,0)
        Q.Cx(2,1)
        Q.x(2)
        Q.sx(2)

    def _applyH(self,
            HamiltonianOperator,
            scaleH=0.5,Nq=4,**kw):
        if Nq==3:
            self.hss = scaleH
            self.use_H = []
            h0 = ['ZII','IZI','IIZ','XII','IXI','IIX']
            h1 = ['XXI','ZZZ','IXX']
            h2 = ['XIX','ZIZ']
            for op in HamiltonianOperator:
                if op.s==h0[0]:
                    self.use_H.append('h0')
                elif op.s==h1[0]:
                    self.use_H.append('h1')
                elif op.s==h2[0]:
                    self.use_H.append('h2')
            get_funcs = {
                    'h0':self._type_H0,
                    'h1':self._type_H1,
                    'h2':self._type_H2,
                    #'h3':self._type_H3,
                    #'h4':self._type_H4,
                    #'h5':self._type_H5,
                    #'h6':self._type_H6,
                    #'h7':self._type_H7,
                    #'h8':self._type_H8,
                    #'h9':self._type_H9,
                    }
            for u in self.use_H[:]:
                self._gates.append(
                        [(HamiltonianOperator,),get_funcs[u]]
                        )

    def _type_H0(self,Q,op):
        h0 = ['ZII','IZI','IIZ','XII','IXI','IIX']
        para = []
        for p in h0:
            for o in op:
                if o.s==p:
                    para.append(o.c.real*self.hss)
        Q.Rz(0,para[0])
        Q.Rz(1,para[1])
        Q.Rz(2,para[2])
        Q.Rx(0,para[3])
        Q.Rx(1,para[4])
        Q.Rx(2,para[5])

    def _type_H1(self,Q,op):
        h1 = ['XXI','ZZZ','IXX']
        para = []
        for p in h1:
            for o in op:
                if o.s==p:
                    para.append(o.c.real*self.hss)
        Q.Cx(0,1)
        Q.Cx(2,1)
        Q.Rx(0, para[0]) #1 XXI
        Q.Rz(1, para[1]) #1 ZZZ
        Q.Rx(2, para[2]) #1 IXX
        Q.Cx(2,1)
        Q.Cx(0,1)


    def _type_H2(self,Q,op):
        h1 = ['XIX','ZIZ']
        para = []
        for p in h1:
            for o in op:
                if o.s==p:
                    para.append(o.c.real*self.hss)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Rx(0,para[0]) #XX
        Q.Rz(2,para[1]) #ZZ
        Q.Cx(0,1)
        Q.Cx(1,2)
        Q.Cx(0,1)
        Q.Cx(1,2)
