from pyscf import gto,scf,mcscf,fci
import numpy as np
np.set_printoptions(precision=5,suppress=True)

mol = gto.Mole()
mol.atom = [['H',(0,0,0)],['H',(2,0,0)]]
mol.verbose=4
mol.build()
hf = scf.RHF(mol)
hf.kernel()
mc = mcscf.CASCI(hf,2,2)
mc.kernel()

d1,d2 = mc.fcisolver.make_rdm12(mc.ci,2,2)
print(d1)
print(d2)

rdm_hqca = np.zeros((2,2,2,2))
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                rdm_hqca[i,k,j,l]+= d2[i,j,k,l]
print(np.reshape(rdm_hqca,(4,4)))

