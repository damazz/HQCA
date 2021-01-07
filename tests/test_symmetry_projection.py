from hqca.quantum_tools import *
from delayed_assert import delayed_assert as da
from hqca.tools.quantum_strings import PauliString as Pauli
from hqca.tools.quantum_strings import FermiString as Fermi
from hqca.tools.quantum_strings import QubitString as Qubit
from hqca.tools._operator import Operator as Op
from hqca.transforms import *
from hqca.symmetry import *
import numpy as np


def test_number_projection():
    a = Op(
            [
                Fermi(coeff=1,ops='+-',indices=[0,1],N=2),
                Fermi(coeff=1,ops='+-',indices=[1,0],N=2),
                ])

    #N = number([0,1],N=2).transform(JordanWigner)
    Nm = operator_to_matrix(N)
    symmetry_projection(a,JordanWigner,N)

def test_spin_squared():
    S2 = total_spin_squared([0,3],[0,1,2,3],[4,5,6,7],N=8)
    print(S2)
    print(S2.transform(JordanWigner))
    print(np.linalg.eigvalsh(operator_to_matrix(S2.transform(JordanWigner))))


#test_number_projection()
test_spin_squared()

