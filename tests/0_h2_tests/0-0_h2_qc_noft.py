from pyscf import gto,scf,mcscf
from hqca.hqca import sp
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9374 +3.5626
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
prog = sp(theory='noft',mol=mol,casci=True)
kw_qc = {
        'Nq':mol.nbas*2,
        'num_shots':4096,
        'entangler_q':'UCC2_1s',
        'spin_mapping':'alternating',
        'depth':1,
        'qc':True,
        'info':None,
        'transpile':True,
        'noise':False,
        'use_radians':True,
        'tomo_extra':'sign_2e_pauli',
        'tomo_approx':'fo',
        'tomo_basis':'no',
        'ansatz':'nat-orb-no',
        'ec_method':'hyperplane',
        'pr_e':2,
        #'error_correction':None
        }
kw_opt = {
        'optimizer':'nevergrad',
        'unity':np.pi/2,
        'nevergrad_opt':'Cobyla',
        'max_iter':5000,
        'conv_crit_type':'MaxDist',
        #'conv_crit_type':'default',
        'conv_threshold':1e-4,
        'N_vectors':2,
        }
orb_opt = {
        'optimizer':'bfgs',
        'conv_threshold':1e-8
        }
prog.set_print(level='diagnostic')
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
prog.update_var(target='orb_opt',**orb_opt)
prog.build()
prog.execute()
