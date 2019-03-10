from pyscf import gto,mcscf,scf
from hqca.main import sp
import sys

mol = gto.Mole()
mol.atom=[
        ['H',(0,0,0)],
        ['H',(1.0,0,0)]]
mol.basis='sto-3g'
#mol.basis = '6-31g'
mol.spin=0
mol.build()
mol.as_Ne = 2
mol.as_No = 2 #spatial

prog = sp(mol,'rdm',calc_E=True,pr_g=2)
kw_qc = {
        'Nqb':mol.as_No*2,
        'tomo_basis':'hada+imag',
        'spin_mapping':'default',
        #'entangler_p':'Uent1_cN',
        #'entangled_pairs':'scheme1_Tavernelli',
        'entangler_p':'UCC1',
        #'entangler_q':'UCC2',
        'entangler_q':'UCC2c',
        'entangled_pairs':'d',
        'ansatz':'ucc',
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
prog.set_print(level='diagnostic_en')
prog.build()
prog.execute()


