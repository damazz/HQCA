import numpy as np
from pyscf import gto,mcscf,scf
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True  )

d1 = np.loadtxt('./en_noisy_ec_cobyla.txt')
d2 = np.loadtxt('./en_noisy_no_ec.txt')
d3 = np.loadtxt('./en_norm_ec.txt')
d4 = np.loadtxt('./en_norm_no_ec.txt')

df = d4[0,:]
ef = d4[2,:]

x1 = [0.525,0.625,0.685,0.735,0.785,0.845,0.915,
        1.000,1.15,1.4,1.7,2.3,3.0]
x1 = [0.525,0.630,0.735,0.840,
        0.95,1.10,1.4,1.8,2.4]
e1 = np.zeros(len(x1))
for n,d in enumerate(x1):
    mol = gto.Mole()
    mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=0
    mol.build()
    hf = scf.RHF(mol)
    hf.kernel()
    mc = mcscf.CASCI(hf,2,2)
    mc.kernel()
    e1[n]=mc.e_tot
#plt.scatter(dist,E_hf,label='HF')
plt.plot(df,ef, label='FCI',zorder=1)
plt.scatter(x1,e1,label='noisy_w_ec',marker='*',color='b',s=50,zorder=2,lw=2)
#plt.axis()
plt.xlabel('H-H Separation',fontsize=12)
plt.ylabel('Energy, H',fontsize=12)
#plt.title('Dissociation Curve of Linear H$_3$')
plt.legend(loc=4)
plt.show()

