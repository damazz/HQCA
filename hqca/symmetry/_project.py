'''
from a given operator, looking to project the operator into the constant
symmetry spaces


'''
from hqca.tools import *
import numpy as np
import sys
from copy import deepcopy as copy
from hqca.tools.quantum_strings import FermiString as Fermi
from hqca.tools.quantum_strings import PauliString as Pauli
import scipy as sp
from hqca.quantum_tools import *


class NewSymmetryProjection:
    def __init__(self,
            op,
            transform,
            quantstore,
            weight='default',
            skip_sz=False,
            verbose=False):
        pass



def _get_matrix_vector(matrix,contracted=True):
    if contracted:
        l = []
        nz = np.nonzero(matrix)
        for i,j in zip(nz[0],nz[1]):
            l.append([(i,j),matrix[i,j]])
        return l
    else:
        return matrix.flatten()

def _get_projection_operators(matrix):
    eigval, eigvec = np.linalg.eigh(matrix)
    p = []
    for s in set(eigval):
        a = np.zeros(eigvec.shape,dtype=np.complex_)
        for i in range(len(eigval)):
            if eigval[i]==s:
                a+= np.dot(np.asmatrix(eigvec[:,i]).T,np.asmatrix(eigvec[:,i]))
        p.append(a)
    return p


def symmetry_projection(
        operator,
        transform,
        symmetries=None,
        ):
    '''
    Given an operator A, and a symmetry (or set of symmetries) with eigenvalues
    defined by s_i, with a transform T, we find a symmetry projected form in the transformed
    basis using eigenvectors of s_i which proejct the operator A.
    '''
    if isinstance(symmetries,type(None)):
        return operator.transform(transform)
    elif isinstance(symmetries,type(Operator())):
        symmetries = [symmetries]
    else:
        print(type(symmetries))
    # now, determine local qubit operators
    # #
    print(symmetries[0])










