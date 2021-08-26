from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString as Qubit
from hqca.operators._operator import Operator as Op
from math import sqrt, pi
import hqca.config as config
config._use_multiprocessing=False

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
