from pyscf import gto,scf,mcscf
from pyscf.tools import molden,cubegen
from hqca.main import sp,scan
from math import pi
import sys
import numpy as np
from functools import reduce
from hqca.tools.bases.ccpvnz import h1,h2,h3,h4,h5

mol = gto.Mole()
d = 0.5
mol.atom = [['H',(0,0,0)],['H',(d,0,0)]]
mol.basis= 'sto-3g'
#mol.basis='6-31g'
#mol.basis='cc-pvdz'
mol.spin=0
mol.verbose=0
mol.build()
mol.as_Ne = 2
mol.as_No = mol.nbas #spatial
#prog = sp(mol,'noft',calc_E=True,pr_g=2)
prog = scan(mol,'noft',calc_E=True,verbose=True)
kw_qc = {
        'Nqb':mol.as_No*2,
        'Nqb_backend':5,
        'num_shots':8192,
        'entangler_q':'UCC2c12',
        'spin_mapping':'default',
        'depth':1,
        'transpile':True,
        'backend_configuration':None,
        'noise':True,
        'noise_model_loc':'20190410_ibmqx4',
        'pr_e':2,
        'qc':True,
        'info':None,
        'transpile':True,
        'use_radians':True,
        'tomo_extra':'sign_2e',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
        #'error_correction':'hyperplane'
        'error_correction':None
        }
kw_opt = {
        'optimizer':'NM',
        'unity':np.pi/4,
        'nevergrad_opt':'OnePlusOne',
        'max_iter':5000,
        'conv_crit_type':'MaxDist',
        'conv_threshold':1e-3,
        'N_vectors':5,
        }
orb_opt = {
        }
prog.set_print(level='default')
#prog.update_var(**{'pr_m':0})
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
prog.update_var(target='orb_opt',**orb_opt)
prog.build()

start = [0]
index = [0]
low = [-pi/180]
high= [pi/180]
ns = [50]
prog.scan('rdm',start,index,high,low,ns,prop='on')



