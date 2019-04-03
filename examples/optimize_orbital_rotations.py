'''
/examples/optimize_orbital_rotations.py

Script to evaluate optimizer efficacy for differen situations. Accomplishes this
by skipping the occupation number optimizer. Can also be used to determine
whether or not a proper minima has been found by the cohsen optimizer.
'''


from pyscf import gto,scf,mcscf
from pyscf.tools import molden,cubegen
from hqca.main import sp
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9374+0.5626
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
#mol.basis='sto-3g'
#mol.basis='6-31g'
mol.basis='cc-pvdz'
mol.spin=0
mol.verbose=0
mol.build()
mol.as_Ne = 2
mol.as_No = mol.nbas #spatial
prog = sp(mol,'noft',calc_E=True,pr_g=2)
kw_qc = {
        'Nqb':mol.as_No*2,
        'num_shots':4096,
        'entangler_q':'UCC2c12',
        'spin_mapping':'default',
        'depth':1,
        'qc':False,
        'use_radians':True,
        'tomo_extra':'sign_2e',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
        'error_correction':'hyperplane'
        }
kw_opt = {
        'optimizer':'NM',
        'unity':np.pi/4,
        'nevergrad_opt':'OnePlusOne',
        'max_iter':5000,
        'conv_crit_type':'MaxDist',
        'conv_threshold':1e-5,
        'N_vectors':5
        }
orb_opt = {
        'conv_threshold':1e-10,
        'conv_crit_type':'default',
        'optimizer':'gpso',
        #'optimizer':'NM',
        'shift':None,
        'gamma':1e-4,
        'particles':30,
        'pso_iterations':50,
        'examples':1,
        'inertia':0.7,
        'max_velocity':0.5,
        'conv_threshold':0.1,
        'unity':np.pi,
        'pr_o':2,
        'accel':[1.0,1.0]
        }
para_631g = [-0.285,0.108,0.809]
para_ccpvdz = [-0.251,0.224 ,0.877,0.779,0.784]

prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
prog.update_var(target='orb_opt',**orb_opt)
prog.set_print(level='diagnostic_orb')
prog.build()
if mol.basis=='6-31g':
    prog.run.single('rdm',para_631g)
elif mol.basis=='ccpvdz':
    prog.run.single('rdm',para_ccpvdz)
prog.run.Store.update_rdm2()
prog.run._find_orb()

#prog.execute()
#prog.analysis()
