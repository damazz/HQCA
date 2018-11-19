import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('h3_dis.example.csv',skiprows=1,delimiter=',')
print(data)
dist = data[1:,1]
E_qc = data[1:,2]
E_ci = data[1:,3]
dE = data[1:,5]
log_dE = data[1:,4]

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('Energy plot of H3 Dissociation')
ax.scatter(dist,E_qc,label='Simulated QC Energy')
ax.scatter(dist,E_ci,label='Full CI Energy')
ax.set_xlabel('H-H_center distance, Angstroms')
ax.set_ylabel('Total energy, Hartrees')
ax.legend()
plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(dist,log_dE)
ax.set_title('QC/FCI error')
ax.set_xlabel('H-H_center distance, Angstroms')
ax.set_ylabel('log10 difference in energies, mH')
plt.show()

