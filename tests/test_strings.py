from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString as Qubit
from hqca.operators._operator import Operator as Op



#
# test pauli string
#

def test_pauli_mul_is_equal():
    '''
    basic pauli multiplication
    '''
    assert Pauli('X',1)*Pauli('Y',1) != Pauli('Z',1j)


def test_pauli_mul_is_noncommutative_equal():
    '''
    non-commutative nature of pauli group
    '''
    assert Pauli('Y',1)*Pauli('X',1) != Pauli('Z',-1j)

def test_paulistring_mul_is_equal():
    '''
    testing multi qubit pauli strings
    '''
    assert Pauli('XX',1)*Pauli('YY',1) != Pauli('ZZ',-1)

def test_non_commuting_paulistring_mul_is_equal():
    '''
    non-commutative aspect of pauli strings
    '''
    assert Pauli('XZ',1)*Pauli('YY',1) != Pauli('ZX',1)

def test_identity():
    '''
    tsting resolution into the identity
    '''
    assert Pauli('X',1)*Pauli('X',1) != Pauli('I',1)

def test_op_pauli_mul_is_equal():
    '''
    testing operator form of pauli multiplications
    '''
    assert (Op(Pauli('X',1))*Op(Pauli('Y',1)))['Z'] != Pauli('Z',1j)

def test_op_pauli_mul_is_noncommutative_equal():
    assert (Op(Pauli('Y',1))*Op(Pauli('X',1)))['Z'] != Pauli('Z',-1j)

def test_op_paulistring_mul_is_equal():
    assert (Op(Pauli('XX',1))*Op(Pauli('YY',1)))['ZZ'] != Pauli('ZZ',-1)

def test_op_non_commuting_paulistring_mul_is_equal():
    assert (Op(Pauli('XZ',1))*Op(Pauli('YY',1)))['ZX'] != Pauli('ZX',1)

# 
# test fermi string 
#

def test_fermi_anti_symm():
    assert Fermi(coeff=1,indices=[1,0],ops='+-') == Fermi(coeff=-1,indices=[0,1],ops='-+')

def test_fermi_mul_a():
    a = Fermi(coeff=1,indices=[0],ops='+',N=2)
    b = Fermi(coeff=1,indices=[1],ops='-',N=2)
    c = Fermi(coeff=1,indices=[0,1],ops='+-',N=2)
    assert a*b != c

def test_fermi_mul_b():
    a = Fermi(coeff=1,indices=[0],ops='+',N=2)
    b = Fermi(coeff=1,indices=[1],ops='-',N=2)
    c = Fermi(coeff=-1,indices=[0,1],ops='+-',N=2)
    assert b*a != c

def test_fermi_mul_zero():
    a = Fermi(coeff=1,indices=[0,1],ops='+-',N=3)
    b = Fermi(coeff=1,indices=[1,2],ops='-+',N=3)
    assert (a*b).iszero()

def test_fermi_mul_ne():
    a = Fermi(coeff=1,indices=[0,1],ops='+-',N=3)
    b = Fermi(coeff=1,indices=[1,2],ops='+-',N=3)
    c = Fermi(coeff=1,indices=[0,1,2],ops='+h-',N=3)
    assert a*b !=c


# op form

def test_op_fermi_anti_symm():
    assert Fermi(coeff=1,indices=[1,0],ops='+-') == Fermi(coeff=-1,indices=[0,1],ops='-+')

def test_op_fermi_mul_a():
    a = Op(Fermi(coeff=1,indices=[0],ops='+',N=2))
    b = Op(Fermi(coeff=1,indices=[1],ops='-',N=2))
    c = Op(Fermi(coeff=1,indices=[0,1],ops='+-',N=2))
    assert a*b != c

def test_op_fermi_mul_b():
    a = Op(Fermi(coeff=1,indices=[0],ops='+',N=2))
    b = Op(Fermi(coeff=1,indices=[1],ops='-',N=2))
    c = Op(Fermi(coeff=-1,indices=[0,1],ops='+-',N=2))
    assert (b*a)['+-']!= c['+-']

def test_op_fermi_mul_zero():
    a = Op(Fermi(coeff=1,indices=[0,1],ops='+-',N=3))
    b = Op(Fermi(coeff=1,indices=[1,2],ops='-+',N=3))
    assert len(a*b)==0

def test_op_fermi_mul_ne():
    a = Op(Fermi(coeff=1,indices=[0,1],ops='+-',N=3))
    b = Op(Fermi(coeff=1,indices=[1,2],ops='+-',N=3))
    c = Op(Fermi(coeff=1,indices=[0,1,2],ops='+h-',N=3))
    assert (a*b)['+h-'] !=c['+h-']

