from pyscf import gto,scf,mcscf
from pyscf.tools import molden,cubegen
from hqca.main import sp
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9374+2.5
mol.atom=[
        ['H',(0,0,0)],
        ['H',(d,0,0)],
        ['H',(-d,0,0)]]
mol.basis='sto-3g'
mol.spin=1
mol.verbose=4
mol.build()
#mol.as_Ne = 3
#mol.as_No = 3
hf = scf.ROHF(mol)
hf.kernel()
hf.analyze()
mc = mcscf.CASCI(hf,3,3)
mc.state_specific_(3)
mc.kernel()
C = hf.mo_coeff
Ci = np.linalg.inv(hf.mo_coeff)

d1 = hf.make_rdm1()
d1 = d1[0]+d1[1]
cubegen.density(mol,'scf_dm_{:.1f}.cube'.format(d),reduce(np.dot, (C,d1,Ci)))


d1 = mc.make_rdm1()
cubegen.density(mol,'fci_dm_{:.1f}.cube'.format(d),reduce(np.dot, (C,d1,Ci)))
'''
molden.from_mo(mol,'hfscf_mo_{:.1f}.molden'.format(d),hf.mo_coeff)
molden.from_mo(mol,'mcscf_mo_{:.1f}.molden'.format(d),mcmo)
'''


sys.exit()
prog = sp(mol,'noft',calc_E=True)
kw = {'pr_g':3}
prog.update_var(**kw)
kw = {
        'pr_m':1,
        'pr_o':0,
        'method':'classical-default',
        'wf_mapping':'zeta'
       }
prog.update_var(main=True,**kw)
kw = {
        'spin_mapping':'unrestricted'
        }
prog.update_var(sub=True,**kw)
prog.execute()
#prog.analysis()










