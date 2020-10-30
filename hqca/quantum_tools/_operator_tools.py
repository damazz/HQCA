from hqca.tools import *
from hqca.transforms import *
import sympy as sy
import numpy as np

'''
tools for manipulating and studying operators
'''

def matrix_to_qubit(mat):
    N = mat.shape[0]
    n = int(np.log2(N))
    basis = ['{:0{}b}'.format(i,n)[::1] for i in range(0,N)]
    op = Operator()
    nz = np.nonzero(mat)
    for i,j in zip(nz[0],nz[1]):
        if abs(mat[i,j])>1e-8:
            e1,e2 = basis[i],basis[j]
            s = ''
            keys = {
                    '0':{
                        '0':'h',
                        '1':'-'},
                    '1':{
                        '0':'+',
                        '1':'p',
                        },
                    }
            for l in range(len(e1)):
                s+= keys[e1[l]][e2[l]]
            op+= QubitString(
                    coeff=mat[i,j],
                    s=s
                    )
    return op

def matrix_to_pauli(mat,operator_type='pauli'):
    N = mat.shape[0]
    n = int(np.log2(N))
    basis = ['{:0{}b}'.format(i,n)[::1] for i in range(0,N)]
    op = Operator()
    nz = np.nonzero(mat)
    for i,j in zip(nz[0],nz[1]):
        if abs(mat[i,j])>1e-8:
            e1,e2 = basis[i],basis[j]
            s = ''
            keys = {
                    '0':{
                        '0':'h',
                        '1':'-'},
                    '1':{
                        '0':'+',
                        '1':'p',
                        },
                    }
            for l in range(len(e1)):
                s+= keys[e1[l]][e2[l]]
            op+= QubitString(
                    coeff=mat[i,j],
                    s=s
                    )
    return op.transform(Qubit)



def operator_to_matrix(op):
    n = len(op[0].s)
    new = np.zeros((2**n,2**n),dtype=np.complex_)
    for o in op:
        if isinstance(o,type(PauliString())):
            temp = Circ(n)
            for q,p in enumerate(o.s):
                if p=='X':
                    temp.x(q)
                elif p=='Y':
                    temp.y(q)
                elif p=='Z':
                    temp.z(q)
        new+= temp.m*o.c
    return new


def partial_trace(obj,**kw):
    if isinstance(obj,type(Operator)):
        return _partial_trace_operator(obj,**kw)
    elif isinstance(obj,type(np.array([]))):
        return _partial_trace_matrix(obj,**kw)
    elif isinstance(obj,type(np.matrix([[]]))):
        return _partial_trace_matrix(obj,**kw)
    elif isinstance(obj,type(sy.Matrix([[]]))):
        return _partial_trace_symbolic_matrix(obj,**kw)
    else:
        print(type(obj))
        print('Not yet supported.')


def _partial_trace_operator(op,qb=[0]):
    new = Operator()
    for o in op:
        if isinstance(o,type(PauliString())):
            new+= o.partial_trace(qb)
    new.clean()
    return new


def _partial_trace_matrix(mat,qb=[0]):
    '''
    trace over the qubits  in qb

    should be listed in reverse order
    '''
    N = mat.shape[0]
    n = int(np.log2(N))
    nd = n-len(qb)
    Nd = len(qb)
    nb = ['{:0{}b}'.format(i,nd)[::1] for i in range(0,2**nd)]
    nbd= {'{:0{}b}'.format(i,nd)[::1]:i for i in range(0,2**nd)}
    keys= {'{:0{}b}'.format(i,n)[::1]:i for i in range(0,2**n)}
    Nb = ['{:0{}b}'.format(i,Nd)[::1] for i in range(0,2**Nd)]
    new = np.zeros((2**nd,2**nd),dtype=np.complex_)
    for l,i in enumerate(nb): #  new
        for m,j in enumerate(nb):
            for basis in Nb:
                ket = '0'*n
                bra = '0'*n
                q,s = 0,0
                for r in range(s,n):
                    if not r in qb:
                        t = i[s] #left
                        u = j[s] #right
                        s+=1
                    else:
                        t = basis[q]
                        u = basis[q]
                        q+=1
                    bra = bra[:r]+t+bra[r+1:] #left
                    ket = ket[:r]+u+ket[r+1:] #right
                ind_bra = keys[bra]
                ind_ket = keys[ket]
                try:
                    new[l,m]+= mat[ind_bra,ind_ket]
                except TypeError:
                    return _partial_trace_symbolic_matrix(mat,qb)
    return new

def _partial_trace_symbolic_matrix(mat,qb=[0]):
    '''
    trace over the qubits  in qb

    should be listed in reverse order
    '''
    N = mat.shape[0]
    n = int(np.log2(N))
    nd = n-len(qb)
    Nd = len(qb)
    nb = ['{:0{}b}'.format(i,nd)[::1] for i in range(0,2**nd)]
    nbd= {'{:0{}b}'.format(i,nd)[::1]:i for i in range(0,2**nd)}
    keys= {'{:0{}b}'.format(i,n)[::1]:i for i in range(0,2**n)}
    Nb = ['{:0{}b}'.format(i,Nd)[::1] for i in range(0,2**Nd)]
    new = sy.zeros(2**nd)
    for l,i in enumerate(nb): #  new
        for m,j in enumerate(nb):
            for basis in Nb:
                ket = '0'*n
                bra = '0'*n
                q,s = 0,0
                for r in range(s,n):
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
                new[l,m]+= mat[ind_ket,ind_bra]
    return new


