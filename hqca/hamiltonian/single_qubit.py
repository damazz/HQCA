from hqca.core import *
import numpy as np
from hqca.tools import *

class SingleQubitHamiltonian(Hamiltonian):
    def __init__(self,sq=True,
            **kw
            ):
        self._order = 1
        self._model = 'sq'
        self._qubOp = ''
        self.No_tot = 1
        self.Ne_tot = 1
        self.real = True
        self.imag = True
        self._en_c = 0
        if sq:
            self._set_operator(**kw)
        else:
            self._set_bloch_sphere(**kw)

    def _set_operator(self,p=0,h=0,c=0,a=0):
        op = Operator()
        #for i,s in zip([p,h,c,a],['p','h','+','-']):
        for i,s in zip([c,a,p,h],['+','-','p','h']):
            temp = QubitOperator(i,indices=[0],sqOp=s)
            temp.generateOperators(Nq=1,real=True,imag=True)
            op+= temp.formOperator()
        self._qubOp = op
        print('Hamiltonian operators: ')
        print(op)
        print('--- --- --- --- ---')
        self._matrix_from_op()

    def _matrix_from_op(self):
        mat = np.zeros((2,2),dtype=np.complex_)
        for i in self._qubOp.op:
            cir = Circ(1)
            if i.p=='X':
                cir.x(0)
            elif i.p=='Y':
                cir.y(0)
            elif i.p=='Z':
                cir.z(0)
            mat+=i.c*cir.m
        self.ef = np.min(np.linalg.eigvalsh(mat))
        self._matrix = np.array([mat])


    @property
    def qubOp(self):
        return self._qubOp

    @qubOp.setter
    def qubOp(self,a):
        self._qubOp = a

    @property
    def matrix(self):
        return self._matrix
    @matrix.setter
    def matrix(self,a):
        self._matrix = a

    @property
    def order(self):
        return self._order
    @order.setter
    def order(self,a):
        self._order = a

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,mod):
        self._model = mod
