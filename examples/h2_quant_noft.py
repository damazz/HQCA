from pyscf import gto,scf,mcscf
from pyscf.tools import molden,cubegen
from hqca.main import sp
import sys
import numpy as np
from functools import reduce
from hqca.tools.bases.ccpvnz import h1,h2,h3,h4,h5

mol = gto.Mole()
d = 0.5 #0.9374 +3.5626
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
#mol.basis='6-31g'
#mol.basis='cc-pvdz'
mol.spin=0
mol.verbose=0
mol.build()
mol.as_Ne = 2
mol.as_No = mol.nbas #spatial
#prog = sp(mol,'noft',calc_E=True,pr_g=2)
filename = '/home/scott/Documents/research/software/hqca/examples/'
filename +='sp-noft_041619-1630.run'
prog =sp(mol,'noft',calc_E=True,verbose=False,restart=filename)
kw_qc = {
        'Nqb':mol.as_No*2,
        'num_shots':4096,
        'entangler_q':'UCC2c12',
        'spin_mapping':'default',
        'depth':1,
        'qc':True,
        'info':None,
        'transpile':True,
        'use_radians':True,
        'tomo_extra':'sign_2e',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
        'error_correction':'hyperplane',
        'pr_e':2,
        #'error_correction':None
        }
kw_opt = {
        #'optimizer':'NM-ng',
        'optimizer':'nevergrad',
        'unity':np.pi,
        'nevergrad_opt':'Cobyla',
        'max_iter':5000,
        'conv_crit_type':'MaxDist',
        #'conv_crit_type':'default',
        'conv_threshold':1e-4,
        'N_vectors':2,
        }
orb_opt = {
        'conv_threshold':1e-8
        }
##prog.set_print(level='default')
###prog.set_print(level='diagnostic_en')
###prog.update_var(**{'pr_m':0})
##prog.update_var(target='qc',**kw_qc )
##prog.update_var(target='opt',**kw_opt)
##prog.update_var(target='orb_opt',**orb_opt)
##prog.build()
prog.execute()
prog.analysis()