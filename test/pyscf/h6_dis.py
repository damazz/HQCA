from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
np.set_printoptions(precision=5,suppress=True)
mol = gto.Mole()
dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237]
dist = np.linspace(0.5,3,11)
print(dist)
dist = [3.0,4.0,5.0]
#dist = [4.0374]
for d in dist:
    mol.atom = [
            ['H',(0,0,d)], 
            ['H',(0,0,-d)], 
            ['H',(d,0,0)], 
            ['H',(-d,0,0)],
            ['H',(0,d,0)], 
            ['H',(0,-d,0)]
            ]
    mol.basis = 'sto-3g'
    mol.spin=0
    mol.verbose=4
    mol.build()

    m = scf.RHF(mol)
    m.max_cycle=500
    m.kernel()
    S = m.get_ovlp()
    mc = mcscf.CASCI(m,6,6)
    mc.verbose=0
    mc.kernel()
    #print('CI coefficients:')
    #print(mc.ci)
    alp,bet = mc.make_rdm1s()
    alp = np.dot(alp,S) 
    bet = np.dot(bet,S)
    rdm1 = mc.make_rdm1()
    rdm1 = np.dot(rdm1,S)
    print('MO coefficients:')
    print(m.mo_coeff)
    print('MO energies: ')
    print(m.mo_energy)
    print('Internuclear Distance: {}'.format(d))
    print('Spatial 1-RDM: ')
    print(rdm1)
    reig,rvec = np.linalg.eig(rdm1)
    print('Eigenvalues of the spatial 1-RDM: ')
    print(np.linalg.eigvalsh(rdm1))
    #print('Alpha 1RDM')
    #print(alp)
    print('Eigenvalues of the 1-RDM:')
    eiga,va = np.linalg.eig(alp)
    eiga.sort()
    print(eiga)
    print('{:.3f} >= {:.3f} ? {}'.format(
                float(eiga[0]+eiga[1]),
                float(eiga[2]),
                eiga[2]<=eiga[1]+eiga[0]
                )
            )
    print('l5+l6-l4: {}'.format(eiga[0]+eiga[1]-eiga[2]))
    #print('Beta 1RDM')
    #print(bet)
    eigb,vb = np.linalg.eig(bet)
    print('Eigenvalues of the 1-RDM:')
    print(np.sort(eigb))
    print('MO coefficients of the natural orbitals:')
    print(mc.mo_coeff)
    print(mc.mo_energy)
    print(rvec)
    print(reig)
    print('\n\n\n')


    #continue
    occslst = fci.cistring._gen_occslst(range(6), 6//2)
    print('Intermolecular H-H0 distance: {}'.format(d))
    print('Full CI Energy: {:.8f} Hartrees'.format(mc.e_tot))
    print('   det-alpha,    det-beta,      CI coefficients')
    for c,ia,ib in mc.fcisolver.large_ci(mc.ci,6,(3,3),tol=0.1, return_strs=False):

        print('     %s          %s          %1.12f' % (ia,ib,c))

