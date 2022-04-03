from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString 
from hqca.operators.quantum_strings import OrderedString as Order
from hqca.operators._operator import Operator as Op
from math import sqrt, pi
import hqca.config as config
config._use_multiprocessing=False
from hqca.transforms import *
from hqca.transforms._mixed_transform import *
from hqca.tools import *

def test_fermi_norm1():
    o = Op()
    o+= Fermi(coeff=1/sqrt(2),indices=[0,1],ops='+-',N=3)
    o+= Fermi(coeff=1/sqrt(2),indices=[1,2],ops='+-',N=3)
    assert abs(o.norm()-1)<1e-10

def test_fermi_norm2():
    o = Op()
    o+= Fermi(coeff=1j/sqrt(2),indices=[0,1],ops='+-',N=3)
    o+= Fermi(coeff=1j/sqrt(2),indices=[1,2],ops='+-',N=3)
    assert abs(o.norm()-1)<1e-10

def test_fermi_norm3():
    o = Op()
    o+= Fermi(coeff=(0.5+0.5j),indices=[0,1],ops='+-',N=3)
    o+= Fermi(coeff=(0.5-0.5j),indices=[1,2],ops='+-',N=3)
    assert abs(o.norm()-1)<1e-10

def test_pauli_norm1():
    o = Op()
    o+= Pauli('XXZ',1/sqrt(2))
    o+= Pauli('XZY',1/sqrt(2))
    assert abs(o.norm()-1)<1e-10

def test_pauli_norm2():
    o = Op()
    o+= Pauli('XXZ',1j/sqrt(2))
    o+= Pauli('XZY',1j/sqrt(2))
    assert abs(o.norm()-1)<1e-10

def test_pauli_norm3():
    o = Op()
    o+= Pauli('XXZ',0.5-0.5j)
    o+= Pauli('XZY',0.5+0.5j)
    assert abs(o.norm()-1)<1e-10

def test_ordered():
    b = QubitString(s='+-',coeff=1)
    a = Order(b)
    c = Fermi(s='-+',coeff=1)
    d = a*c
    A = Order(c)
    
    B = Qubit(b)
    C = JordanWigner(c)
    T = get_mixed_transform(qubit_transform=Qubit,fermi_transform=JordanWigner)
    y = Op()
    y+=A
    y+=a 
    print(y)
    #    print(d)
    #    print(T(d))
    #    print('')
    #    print(B*C)
    #

def test_matrix():
    a = Operator()+Fermi(s='+-+-',coeff=1)
    A = a.transform(JordanWigner)
    Am = operator_to_matrix(A)
    b = PauliString('XXXX',1)
    mat = -0.5*operator_to_matrix(PauliString('XXXX',1))
    mat+= 0.5*operator_to_matrix(PauliString('XYYX',1))
    print(mat)

test_matrix()

