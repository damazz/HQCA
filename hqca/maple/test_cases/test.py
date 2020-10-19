from hqca.maple._purify import *
import sys
import numpy as np
import pickle 
from hqca.tools import *


names = [
        'tmp7bh4a4ao',
        'tmpaibijpo1',
        'tmpfggavzfr',
        'tmpmy63gano',
        'tmppuyaelyy',
        'tmpwr9uncco',
        'tmpzrqj0nat'
        ]
class QuantStore:
    def __init__(self):
        self.path_to_maple = '/home/scott/maple2020/bin/maple'
        self.spin_rdm = True
        self.groups = [[0,1,2,3],[4,5,6,7]]
        self.Ne = 4
        self.Ne_alp =2
        self.Ne_bet = 2
        self.No_as = 4


for name in names:
    pure = purify_rdm(name,QuantStore())
    pure.analysis()
    pure.get_spin_properties()
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


