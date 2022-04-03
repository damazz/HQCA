from hqca.tools import *
from delayed_assert import delayed_assert as da
from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString as QubitString
from hqca.operators.quantum_strings import OrderedString
from hqca.operators._operator import Operator as Op
from hqca.transforms import *
import numpy as np
import hqca.config as config
config._use_multiprocessing=False

from hqca.tomography.__symmetry_project import *


def test_qubit_symm():
    o = Operator()
    o+= FermiString(coeff=1,ops='-++-',indices=[0,1,2,3],N=4)
    o+= FermiString(coeff=1,ops='+--+',indices=[0,1,2,3],N=4)
    print(o)
    print(o.transform(JordanWigner))
    s = SymmetryProjection(
        o,JordanWigner,
        alp=[0,1,2,3],bet=[],
    )
    print('')
    print(s.qubOp)
#
#test_qubit_symm()

def test_mixed_symm():

    f1= PartialJordanWigner(FermiString(coeff=1,ops='+-',indices=[0,3,],N=8))
    f1d= PartialJordanWigner(FermiString(coeff=+1,ops='+-',indices=[1,2],N=8))


    #f2 = FermiString(coeff=+1,ops='+-',indices=[2,1],N=8)
    #f2d= FermiString(coeff=+1,ops='+-',indices=[3,0,],N=8)

    op = Operator()+f1*f1d


    s = SymmetryProjection(
        op,Qubit,
        alp=[0,1,2,3],bet=[4,5,6,7],
    )
    #print('')
    print(s.qubOp)

test_mixed_symm()
#def test_number_projection():
#    a = Op(
#            [
#                Fermi(coeff=1,ops='+-',indices=[0,1],N=2),
#                Fermi(coeff=1,ops='+-',indices=[1,0],N=2),
#                ])
#
#    N = number([0,1],N=2).transform(JordanWigner)
#    Nm = operator_to_matrix(N)
#    symmetry_projection(a,JordanWigner,N)
#
#def test_spin_squared():
#    S2 = total_spin_squared([0,3],[0,1,2,3],[4,5,6,7],N=8)
#    print(S2)
#    print(S2.transform(JordanWigner))
#    print(np.linalg.eigvalsh(operator_to_matrix(S2.transform(JordanWigner))))
#
#

