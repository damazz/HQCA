from hqca.hamiltonian import *
from _generic import *
from delayed_assert import delayed_assert as da
import numpy as np
import pickle
import hqca.config as config
config._use_multiprocessing=False

def test_molecular():
    ham = generic_molecular_hamiltonian()
    print(ham.K2)
    print(np.reshape(ham.K2,(16,16)))
    s = ''
    for i in range(4):
        for j in range(4):
            s+= '{}{} '.format(i,j)
    print(s)
    for i in range(4):
        print('k2')
        print(0,i,0,i,ham.K2[0,i,0,i])
        print(i,0,i,0,ham.K2[i,0,i,0])
        print('ints')
        print(0,i,0,i,ham.ints_2e[0,i,0,i])
        print(i,0,i,0,ham.ints_2e[i,0,i,0])
    e0 = -0.783792654277353
    da.expect(abs(ham.e0-e0)<1e-10)
    # check initializiation

    # check tr k1 k2 

    d1 = np.load('./store/d1.npy',allow_pickle=True)
    d2 = np.load('./store/d2.npy',allow_pickle=True)

    en = np.dot(d1,ham.ints_1e).trace().real
    en+= 0.5*np.dot(
            np.reshape(d2,(16,16)),
            np.reshape(ham.ints_2e,(16,16))
            ).trace().real
    en+= ham._en_c
    da.expect(abs(en-e0)<1e-10)
    da.assert_expectations()

def test_fermionic():
    ham = generic_fermionic_hamiltonian()
    e0 = -0.783792654277353
    # check initializiation
    # check tr k1 k2 
    d1 = np.load('./store/d1.npy',allow_pickle=True)
    d2 = np.load('./store/d2.npy',allow_pickle=True)

    en = np.dot(d1,ham.ints_1e).trace().real
    en+= 0.5*np.dot(
            np.reshape(d2,(16,16)),
            np.reshape(ham.ints_2e,(16,16))
            ).trace().real
    en+= ham._en_c
    da.expect(abs(en-e0)<1e-10)
    da.assert_expectations()



test_fermionic()
