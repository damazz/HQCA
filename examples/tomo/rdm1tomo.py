from pyscf import gto,scf,mcscf
from math import pi
from pyscf.tools import molden,cubegen
from hqca.main import sp,scan
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9374+0.5626
mol.atom=[
        ['H',(0,0,0)],
        ['H',(d,0,0)]]
mol.basis='sto-3g'
#mol.basis='6-31g'
mol.spin=0
mol.verbose=0
mol.build()
mol.as_Ne = 2
mol.as_No = mol.nbas #spatial
prog = scan(mol,'noft',calc_E=True,verbose=False)

kw_qc = {
        'Nqb':mol.as_No*2,
        'Nqb_backend':5,
        'num_shots':2048,
        'entangler_q':'UCC2c12v2',
        'spin_mapping':'default',
        'depth':1,
        'transpile':True,
        'noise':True,
        'noise_model_loc':'20190418_ibmqx2',
        'use_radians':True,
        'tomo_extra':'sign_2e_pauli',
        'tomo_basis':'no',
        'pr_q':3,
        'tomo_approx':'fo', #'fo','so','full'
        'ansatz':'nat-orb-no',
        #'error_correction':'hyperplane'
        'error_correction':None,
        }
kw_opt = {
        'optimizer':'nevergrad',
        'unity':np.pi/2,
        'nevergrad_opt':'Cobyla',
        'max_iter':500,
        'conv_crit_type':'MaxDist',
        'N_vectors':5
        }

prog.set_print(level='diagnostic')
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
#
prog.build()
for i in np.linspace(pi/4,pi/2,2):
    prog.update_rdm([i])
    print('')
    print('')

#prog.update_rdm([0])
#prog.scan('rdm',start,index,high,low,ns)

