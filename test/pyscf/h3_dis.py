from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys

mol = gto.Mole()
dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237]
#dist = [4.0374]
for d in dist:
    mol.atom = [['H0',(0,0,0)], ['H',(0,0,-d)], ['H',(0,0,d)]]
    mol.basis = 'ccpvdz'
    mol.spin=3
    mol.verbose=1
    mol.build()

    m = scf.ROHF(mol)
    m.max_cycle=100
    m.kernel()
    mc = mcscf.CASSCF(m,3,3)
    mc.verbose=4
    mc.kernel()
    #print('CI coefficients:')
    #print(mc.ci)


    occslst = fci.cistring._gen_occslst(range(3), 3//2)
    print('Intermolecular H-H0 distance: {}'.format(d))
    print('Full CI Energy: {:.8f} Hartrees'.format(mc.e_tot))
    print('   det-alpha,    det-beta,      CI coefficients')
    for c,ia,ib in mc.fcisolver.large_ci(mc.ci,3,(3,0),tol=0.01, return_strs=False):

        print('     %s          %s          %1.12f' % (ia,ib,c))

