from pyscf import gto,scf,mcscf
from hqca.hqca import sp
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 1.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.charge=1
mol.verbose=0
mol.build()
prog = sp(theory='acse',mol=mol,casci=True,max_iter=50,time=0.1)
kw_qc = {
        'Nq':mol.nbas*2,
        'num_shots':8192,
        'entangler_q':'UCC2_2s',
        'spin_mapping':'default',
        'method':'qq-acse2',
        'depth':1,
        'qc':True,
        'info':None,
        'transpile':True,
        'noise':False,
        'use_radians':True,
        'pr_e':2,
        'ec_post':False,
        }
#prog.set_print(level='diagnostic')
prog.update_var(target='qc',**kw_qc )
prog.build()
prog.run()
