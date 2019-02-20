from pyscf import gto,mcscf,scf
from hqca.main import sp

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


prog = sp(mol,'rdm',calc_E=True)
kw_qc = {
        'Nqb':mol.as_No*2,
        'tomo_basis':'hada+imag',
        'spin_mapping':'spin-free',
        'entangler_p':'Uent1_cN',
        'entangled_pairs':'scheme1_Tavernelli',
        'pr_q':0,
        'depth':6
        }
kw_opt = {'optimizer':'NM'}
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
prog.update_var(target='global',**{'pr_g':3,'pr_m':2})
prog.execute()


