from hqca.core import *
import numpy as np
from hqca.tools import *

class ThreeQubitHamiltonian(Hamiltonian):
    def __init__(self,sq=True,
            fermi=False,
            en_c=0,
            real=True,
            imag=False,
            **kw
            ):
        self._order=3
        self._model = '3q'
        self._qubOp = ''
        self.real = real
        self.imag = imag
        self._en_c = en_c
        if sq:
            if not fermi:
                self._set_operator(**kw)
            else:
                self._fermi_to_bosonic(**kw)
        else:
            if not fermi:
                self._op_from_matrix(**kw)
            else:
                self._op_from_ints(**kw)

    def _set_operator(self,p=0,h=0,c=0,a=0):
        op = Operator()
        for i,s in zip([p,h,c,a],['p','h','+','-']):
            temp = QubitOperator(i,indices=[0],sqOp=s)
            temp.generateOperators(Nq=1,real=True,imag=True)
            op+= temp.formOperator()
        self._qubOp = op
        self._matrix_from_op()

    def _op_from_matrix(self,matrix):
        b = ['{:0{}b}'.format(i,3)[::1] for i in range(8)]
        m2b = {i:b[i] for i in range(8)}
        def populate(i,j):
            maps = {
                    '0':{ #ket
                        '0':'h',
                        '1':'+'},
                    '1':{ #bra
                        '0':'-',
                        '1':'p'
                        }}
            I,J = m2b[i],m2b[j]
            code = ''
            for z in range(3):
                code+= maps[J[z]][I[z]]
            return code
    
        self.sq_map = {}# tuple, code
        for i in range(8):
            for j in range(8):
                self.sq_map[(i,j)]=populate(i,j)
        ops = np.nonzero(matrix)
        op = Operator()
        for ind in np.transpose(ops):
            s = self.sq_map[tuple(ind)]
            temp = QubitOperator(matrix[tuple(ind)],indices=[0,1],sqOp=s)
            temp.generateOperators(Nq=2,real=True,imag=True)
            op+= temp.formOperator()
        self._qubOp = op
        self._matrix = np.array([matrix])

    def _matrix_from_op(self):
        mat = np.zeros((8,8),dtype=np.complex_)
        for i in self._qubOp.op:
            cir = Circ(3)
            for n in [0,1]:
                if i.p[n]=='X':
                    cir.x(n)
                elif i.p[n]=='Y':
                    cir.y(n)
                elif i.p[n]=='Z':
                    cir.z(n)
            mat+=i.c*cir.m
        self.ef = np.min(np.linalg.eigvalsh(mat))+self._en_c
        self._matrix = np.array([mat])
        print(self._qubOp)
        print(self._matrix)

    def _fermi_to_bosonic(self,ferOp,
            mapOrb,
            mapQub=None,
            ):
        print('Generating bosonic operators from fermionic operators:')
        # mapOrb : spin to spatial? 
        Op = Operator()
        for item in ferOp.op:
            if item.opType=='no':
                check = mapQub[item.qInd[0]]==0
                sq = 'h'*(check)+'p'*(1-check)
                newInd = [mapOrb[item.qInd[0]]]
                newOp = QubitOperator(
                        coeff=item.c,
                        indices=newInd,
                        sqOp=sq)
                newOp.generateOperators(Nq=3,real=True,imag=True)
                Op+= newOp.formOperator()
            elif item.opType=='nn':
                c1 = mapQub[item.qInd[0]]==0
                c2 = mapQub[item.qInd[2]]==0
                sq1 = 'h'*(c1)+'p'*(1-c1)
                sq2 = 'h'*(c2)+'p'*(1-c2)
                newInd = [mapOrb[item.qInd[0]],mapOrb[item.qInd[2]]]
                newOp = QubitOperator(
                        coeff=item.c,
                        indices=newInd,
                        sqOp=sq1+sq2)
                newOp.generateOperators(Nq=3,real=True,imag=True)
                Op+= newOp.formOperator()
            elif item.opType=='de':
                conj = {'++':'--','+-':'-+','-+':'+-','--':'++'}
                q = [mapQub[i] for i in item.qInd]
                o = [mapOrb[i] for i in item.qInd]
                sq = ''
                for i in range(2):
                    s = ''
                    for j in range(4):
                        if o[j]==i:
                            s+= item.qOp[j]
                    sq+=  int('+-'==s)*'-'+int(1-('+-'==s))*'+'
                new = []
                for i in item.qInd:
                    if mapOrb[i] in new:
                        pass
                    else:
                        new.append(mapOrb[i])
                if len(new)==2:
                    pass
                else:
                    continue
                c1 = item.qOp[0:2]=='+-'
                newOp = QubitOperator(
                        coeff=item.c,
                        indices=new,
                        sqOp=sq)
                newOp.generateOperators(Nq=3,real=True,imag=True)
                Op+= newOp.formOperator()
                newOp = QubitOperator(
                        coeff=item.c,
                        indices=new,
                        sqOp=conj[sq])
                newOp.generateOperators(Nq=3,real=True,imag=True)
                Op+= newOp.formOperator()
            else:
                print('Unused operator: ')
                print(item)
                continue
        self._qubOp = Op
        self._matrix_from_op()


    def _op_from_ints(self,
            ints,
            mapOrb=None,
            mapQub=None,
            mapSqOp=None,
            ints_thresh=1e-14,
            **kw):
        sys.exit('Can\'t get operators from integrals. Use ops!')


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


