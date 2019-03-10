from pyscf import gto,scf,mcscf
from pyscf.tools import molden,cubegen
from hqca.main import sp
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9374+0.0626
mol.atom=[
        ['H',(0,0,0)],
        ['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
mol.as_Ne = 2
mol.as_No = 2 #spatial
prog = sp(mol,'noft',calc_E=True,pr_g=2)

kw_qc = {
        'Nqb':mol.as_No*2,
        'entangler_q':'UCC2',
        'spin_mapping':'default',
        'depth':1,
        'use_radians':True
        }
kw_opt = {
        'optimizer':'nevergrad',
        'nevergrad_opt':'Cobyla',
        'max_iter':500,
        'conv_crit_type':'MaxDist'
        }
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
#
prog.set_print(level='default')
prog.build()
prog.execute()




