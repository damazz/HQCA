import numpy as np
from hqca.core import *
from hqca.core.primitives import *
from hqca.tools import *
import sys

class Line4Q(Instructions):
    '''
    type 1, 2 and 3
    '''
    def __init__(self,operator,
            Nq,propagate=False,**kw):
        self.use = []
        self._gates = []
        a1 = ['YZZI','YIII','YZIZ','YIZZ']
        a2 = ['ZYIZ','IYZZ','ZYZI','IYII']
        b1 = ['IXYX','IXXY','IYXX','IYYY']
        b2 = ['XIXY','YIYY','XIYX','YIXX']
        c1 = ['XZZY','XIIY','IYXI','IXYI']
        for op in operator:
            print(op)
            if op.s==a1[0]:
                self.use.append('a1')
            elif op.s==a2[0]:
                self.use.append('a2')
            elif op.s==b1[0]:
                self.use.append('b1')
            elif op.s==b2[0]:
                self.use.append('b2')
            elif op.s==c1[0]:
                self.use.append('c1')
            elif op.s not in (a1+a2+b1+b2+c1):
                print('Not recognized.')
            #if len(self.use)==5:
            #    break
        get_funcs = {
                'a1':self._type_A1,
                'a2':self._type_A2,
                'b1':self._type_B1,
                'b2':self._type_B2,
                'c1':self._type_C1,
                }
        for u in self.use:
            self._gates.append(
                    [(operator,),get_funcs[u]]
                    )
        print(self.use)

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

    def _Y1(self,Q,op):
        paulis = ['YIII','YZZI','YZIZ','YIZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)

    def _type_A1(self,Q,op):
        '''
        connectvitiy?

        0-1-2-3 (old)
        ----> 

        1-2-3-0
        '''
        paulis = ['YIII','YZZI','YZIZ','YIZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(0)
        Q.h(0)
        Q.si(0)
        Q.Rz(0,-para[0])


        Q.Cx(2,3)
        Q.Cx(1,2)
        Q.Cx(0,3)
        Q.Cx(3,0)
        Q.Rz(3,-para[3]) #YZZI
        Q.Cx(2,3)
        Q.Rz(3,-para[2]) #YZIZ
        Q.Cx(3,0)
        Q.Rz(0,-para[1]) #YIZZ
        Q.Cx(3,0)
        Q.Cx(2,3)
        Q.Cx(3,0)
        Q.Cx(0,3)
        Q.Cx(1,2)
        Q.Cx(2,3)

        Q.s(0)
        Q.h(0)
        Q.si(0)

    def _type_A2(self,Q,op):
        '''
        1A with a swap between (0,1)

        but also had to swap 2-3 because...yeah
        '''
        #paulis = ['IYII','ZYZI','ZYIZ','IYZZ']
        paulis = ['IYII','ZYZI','ZYIZ','IYZZ']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(1)
        Q.h(1)
        Q.si(1)
        Q.Rz(1,-para[0])
        Q.Cx(2,0)
        Q.Cx(3,2)
        Q.Cx(1,0)
        Q.Cx(0,1)
        Q.Rz(0,-para[1]) #ZYZI
        Q.Cx(2,0)
        Q.Rz(0,-para[2]) #ZYIZ
        Q.Cx(0,1)
        Q.Rz(1,-para[3]) #IYZZ
        Q.Cx(0,1)
        Q.Cx(2,0)
        Q.Cx(0,1)
        Q.Cx(1,0)
        Q.Cx(3,2)
        Q.Cx(2,0)
        Q.s(1)
        Q.h(1)
        Q.si(1)

    def _type_B1(self,Q,op):
        paulis = ['IXXY','IXYX','IYXX','IYYY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
                    break
        Q.s(1)
        Q.h(1)
        Q.si(1)
        Q.s(2)
        Q.s(3)
        Q.h(3)
        Q.Cx(1,2)
        Q.s(2)
        Q.Cx(3,2)
        #Q.Ry(3,-iyxx)
        Q.Ry(3,-para[2])
        Q.Cx(2,1)
        Q.si(1)
        Q.Rx(1,para[0])
        #Q.Rx(1,ixxy)
        Q.Cx(3,2)
        #Q.Ry(3,-ixyx)
        Q.Ry(3,-para[1])
        Q.s(2)
        Q.s(1)
        Q.Cz(2,1)
        Q.Cx(3,2)
        Q.Cx(2,1) #!
        #Q.Rx(2,-iyyy)
        Q.Rx(2,-para[3])
        Q.s(2)
        Q.Cx(1,2)
        Q.Cx(3,2)
        Q.si(3)
        Q.h(3)
        Q.si(3)
        Q.z(1)
        Q.h(1)
        Q.si(1)
        Q.z(2)

    def _type_B2(self,Q,op):
        ''' 
        swap with (0,1) 
        '''
        paulis = ['XIXY','XIYX','YIXX','YIYY']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.s(0)
        Q.h(0)
        Q.si(0)
        Q.s(2)
        Q.s(3)
        Q.h(3)
        Q.Cx(0,2)
        Q.s(2)
        Q.Cx(3,2)
        #Q.Ry(3,-iyxx)
        Q.Ry(3,-para[2])
        Q.Cx(2,0)
        Q.si(0)
        Q.Rx(0,para[0])
        #Q.Rx(1,ixxy)
        Q.Cx(3,2)
        #Q.Ry(3,-ixyx)
        Q.Ry(3,-para[1])
        Q.s(2)
        Q.s(0)
        Q.Cz(2,0)
        Q.Cx(3,2)
        Q.Cx(2,0) #!
        #Q.Rx(2,-iyyy)
        Q.Rx(2,-para[3])
        Q.s(2)
        Q.Cx(0,2)
        Q.Cx(3,2)
        Q.si(3)
        Q.h(3)
        Q.si(3)
        Q.z(0)
        Q.h(0)
        Q.si(0)
        Q.z(2)


    def _type_C1(self,Q,op):
        ''' 
        '''
        paulis = ['XZZY','XIIY','IYXI','IXYI']
        para = []
        for p in paulis:
            for o in op:
                if o.s==p:
                    para.append(o.c.imag)
        Q.h(0)
        Q.si(3)
        Q.h(3)
        
        Q.Cx(0,3)
        Q.Cx(3,2)
        Q.Cx(2,1)
        # 
        Q.Rz(1,para[0])
        Q.Rz(3,para[1])
        Q.Cx(2,1)
        Q.Cx(3,2)
        Q.Cx(0,3)
        
        Q.z(2)
        Q.z(1)
        
        Q.h(1)
        Q.Cx(1,2)
        Q.Ry(1,-para[2])
        Q.Ry(2,para[3])
        Q.Cx(1,2)
        Q.h(1)

        Q.z(1)
        Q.z(2)
        Q.h(3)
        Q.s(3)
        Q.h(0)

