#!/usr/bin/python3
'''
./examples/H3.py 

File to run a dissociation curve with 
'''
import numpy as np
import sys,os 
from importlib import reload
import subprocess
os.chdir('/home/scott/Documents/research/3_vqa/hqca')
sys.path.insert(1,'{}'.format(os.getcwd()))
from examples import func as fx
Np = 10
E_qc  = np.zeros(Np)
E_fci = np.zeros(Np)
bond_dist = np.zeros(Np)
mol_loc = []
input_file = './examples/input/h3.txt'
for i in range(0,Np):
    mol_loc.append('./examples/mol/h3_{}.py'.format(i))

for i in range(0,Np):
    fx.change_config(input_file,mol_loc[i])
    #print(mol_loc[i])
    #from main import pre
    if i==0:
        import main
    else:
        reload(main)
    E_qc[i] = main.Store.energy_best
    E_fci[i] = main.E_fci
    dist = float(
            main.mol.mol.atom.split(';')[2].split(' ')[-1])
    bond_dist[i]=dist
    bond_dist[i]=dist
    from main import pre
    from main import chem

print(E_qc)
print(E_fci)

print('----------')
print('----------')
print('----------')
print('----------')
print('Done with measurements.')
import matplotlib.pyplot as plt
import datetime
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(121)
ax1.set_title('Energy plot of H3 Dissociation')
ax1.scatter(bond_dist,E_qc,label='Simulated QC Energy',marker='x')
ax1.scatter(bond_dist,E_fci,label='Full CI Energy',c='r',marker='o')
ax1.set_xlabel('H-H_center distance, Angstroms')
ax1.set_ylabel('Total energy, Hartrees')
ax1.legend()

dE = E_qc-E_fci
log_dE = np.log10(dE)
ax2 = fig.add_subplot(122)
ax2.scatter(bond_dist,log_dE)
ax2.set_title('QC - FCI error')
ax2.set_xlabel('H-H_center distance, Angstroms')
ax2.set_ylabel('log10 difference in energies, H')
plt.savefig('./examples/test.png',dpi=200)
plt.show()






