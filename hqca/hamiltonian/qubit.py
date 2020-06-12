from hqca.core import *
import sys
from hqca.tools import *
import numpy as np

class QubitHamiltonian(Hamiltonian):
    def __init__(self,
            qubits=3,
            operator='pauli',
            order=3,
            **kw,
            ):
        pass
        self._en_c = 0
        self._order=order
        self._model = 'q'
        self._qubOp = ''
        self.N = qubits
        self.p = order
        if operator=='pauli':
            self._matrix_from_pauli(**kw)
            pass
        elif operator=='sq':
            pass

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

    def _matrix_from_pauli(self,pauli):
        mat = np.zeros((2**self.N,2**self.N),dtype=np.complex_)
        for i in pauli.op:
            cir = Circ(self.N)
            for n,p in enumerate(i.s):
                if p=='X':
                    cir.x(n)
                elif p=='Y':
                    cir.y(n)
                elif p=='Z':
                    cir.z(n)
            mat+=i.c*cir.m
        if self.p==self.N:
            self._matrix = np.array([mat])
        else:
            self._generate_reduced_hamiltonian(mat)
        self.ef = np.min(np.linalg.eigvalsh(mat))+self._en_c
        print(self._qubOp)
        #print(self._matrix)

    def _generate_reduced_hamiltonian(self,mat):
        # generate pairs, triplets, etc. 
        # secretly, we are just going to use a qRDM object, because they
        # actually have the correct formatting here
        rhomat = DensityMatrix(size=self.N)
        rhomat.rho = mat
        new = qRDM(
                order=self.p,
                Nq=self.N,
                state='empty'
                )
        new.from_density_matrix(rhomat)
        self._matrix = new.rdm*(1/(2**(self.N-self.p)))

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

