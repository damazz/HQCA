from pyscf import gto,scf,mcscf
from pyscf.tools import molden,cubegen
from hqca.main import sp
import sys
import numpy as np
from functools import reduce

kw_qc = {
        'Nqb_backend':5,
        'num_shots':4096,
        'entangler_q':'UCC2c12v2',
        'spin_mapping':'default',
        'depth':1,
        'transpile':True,
        'backend_configuration':None,
        'noise':True,
        'noise_model_loc':'20190410_ibmqx4',
        'qc':True,
        'pr_e':2,
        'info':None,
        'use_radians':True,
        'tomo_extra':'sign_2e',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
        'error_correction':'hyperplane'
        #'error_correction':None
        }
kw_opt = {
        'optimizer':'nevergrad',
        'unity':np.pi/4,
        'nevergrad_opt':'Cobyla',
        'max_iter':5000,
        'conv_crit_type':'MaxDist',
        'conv_threshold':1e-4,
        'N_vectors':2,
        }
orb_opt = {
        }
dist = np.arange(0.5,1.05,0.05)
E = np.zeros((2,len(dist)))
for n,d in enumerate(dist):
    mol = gto.Mole()
    mol.atom = [['H',(0,0,0)],['H',(d,0,0)]]
    mol.basis= 'sto-3g'
    mol.spin=0
    mol.verbose=0
    mol.build()
    mol.as_Ne = 2
    mol.as_No = mol.nbas #spatial
    kw_qc['Nqb']=mol.as_No*2
    prog = sp(mol,'noft',calc_E=True,verbose=True)
    print('#######')
    print('Distance: {}'.format(d))
    print('#######')
    prog.set_print(level='diagnostic')
    prog.update_var(target='qc',**kw_qc )
    prog.update_var(target='opt',**kw_opt)
    prog.update_var(target='orb_opt',**orb_opt)
    prog.build()
    prog.execute()
    prog.analysis()
    E[0,n]=prog.run.total.crit
    E[1,n]=prog.Store.kw['e_fci']


tot = np.zeros((3,len(dist)))
tot[0,:]=dist[:]
tot[1:,:]=E[:,:]
print('Distance: ')
print(dist)
print('Energies: ')
print(E)
np.savetxt('en_noisy_ec_cobyla.txt',tot)




