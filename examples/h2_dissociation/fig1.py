import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True  )

d1 = np.loadtxt('./en_noisy_ec.txt')
d2 = np.loadtxt('./en_noisy_no_ec.txt')
d3 = np.loadtxt('./en_norm_ec.txt')
d4 = np.loadtxt('./en_norm_no_ec.txt')

x1 = d1[0,:]
x2 = d2[0,:]
x3 = d3[0,:]
x4 = d4[0,:]

e1 = d1[1,:]
e2 = d2[1,:]
e3 = d3[1,:]
e4 = d4[1,:]

ef = d3[2,:]
ef2= d1[2,:]
df = d3[0,:]
print(d4,e4)

dif1 = e1-ef2
dif2 = e2-ef2
dif3 = e3-ef
dif4 = e4-ef


#plt.scatter(dist,E_hf,label='HF')
plt.plot(df,ef, label='FCI',zorder=1)
plt.scatter(x4,e4,label='stoc_no_ec',marker='o',color='k',s=50,zorder=2,lw=2)
plt.scatter(x3,e3,label='stoc_w_ec',marker='x',color='g',s=50,zorder=2,lw=2)
plt.scatter(x2,e2,label='noisy_no_ec',marker='d',color='r',s=50,zorder=2,lw=2)
plt.scatter(x1,e1,label='noisy_w_ec',marker='*',color='b',s=50,zorder=2,lw=2)
#plt.axis()
plt.xlabel('H-H Separation',fontsize=12)
plt.ylabel('Energy, H',fontsize=12)
#plt.title('Dissociation Curve of Linear H$_3$')
plt.legend(loc=4)
plt.show()
sys.exit()


a1 = plt.axes([0.55,0.55,.30,.25],facecolor=(1,1,0.7))
a1.set_yscale('log')
plt.text(1.2,1.9,'``Chemical Accuracy\"',fontsize=7)
plt.scatter(dist,dE)
plt.ylim((5e-3,5e0))
plt.plot(dist,dE_chem,linestyle='dashed')

plt.ylabel('Log$_{10}$ Error, mH',fontsize=9)

plt.show()
