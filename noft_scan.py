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
#mol.basis='sto-3g'
mol.basis='6-31g'
mol.spin=0
mol.verbose=0
mol.build()
mol.as_Ne = 2
mol.as_No = mol.nbas #spatial
prog = scan(mol,'noft',calc_E=True,pr_g=2)

kw_qc = {
        'Nqb':mol.as_No*2,
        'num_shots':4096,
        'entangler_q':'UCC2c12',
        'spin_mapping':'default',
        'depth':1,
        'use_radians':True,
        'tomo_extra':'sign_2e',
        'ansatz':'nat-orb-no',
        'error_correction':'hyperplane'
        }
kw_opt = {
        'optimizer':'nevergrad',
        'unity':np.pi/2,
        'nevergrad_opt':'Cobyla',
        'max_iter':500,
        'conv_crit_type':'MaxDist',
        'N_vectors':5
        }
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
#
prog.set_print(level='default')
prog.build()
prog.scan(-pi/2,pi/2,5,-pi/2,pi/2,5)

