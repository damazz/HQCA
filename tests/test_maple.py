from hqca.maple import *
import sys
import numpy as np
import pickle
import hqca.config as config
config._use_multiprocessing=False


class QuantStore:
    def __init__(self):
        #self.path_to_maple = '/home/scott/maple2020/bin/maple'
        self.spin_rdm = True
        self.groups = [[0,1,2,3],[4,5,6,7]]
        self.Ne = 4
        self.Ne_alp =2
        self.Ne_bet = 2
        self.No_as = 4


def beta_test_maple_spin_purify():
    name = 'temp{}'.format(1)
    pure = purify_rdm(rdm_location+name,QuantStore())
    pure.analysis()
    #pure.get_spin_properties()
    pure.contract()
    print('Sz: {:.8f}'.format(np.real(pure.sz)))
    print('S2: {:.8f}'.format(np.real(pure.s2)))
    print('N:  {:.8f}'.format(np.real(pure.trace())))
    print('---------------------')
    print('---------------------')
    print(np.linalg.eigvalsh(pure.rdm))
    print('---------------------')
    print('---------------------')
    print('\n\n')
    print('---------------------')
    print('---------------------')


rdm_location = './store/maple_rdms/'

'''
for i in range(1):
    name = 'temp{}'.format(i+1)
    pure = purify_rdm(rdm_location+name,QuantStore())
    pure.analysis()
    #pure.get_spin_properties()
    pure.contract()
    print('Sz: {:.8f}'.format(np.real(pure.sz)))
    print('S2: {:.8f}'.format(np.real(pure.s2)))
    print('N:  {:.8f}'.format(np.real(pure.trace())))
    print('---------------------')
    print('---------------------')
    print(np.linalg.eigvalsh(pure.rdm))
    print('---------------------')
    print('---------------------')
    print('\n\n')
    print('---------------------')
    print('---------------------')

    #assert Pauli('X',1)*Pauli('Y',1) != Pauli('Z',1j)
'''
