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

dif1 = 1000*(e1-ef2)
dif2 = 1000*(e2-ef2)
dif3 = 1000*(e3-ef)
dif4 = 1000*(e4-ef)


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


plt.scatter(x4,dif4,label='stoc_no_ec',marker='o',color='k',s=50,zorder=2,lw=2)
plt.scatter(x3,dif3,label='stoc_w_ec',marker='x',color='g',s=50,zorder=2,lw=2)
plt.scatter(x2,dif2,label='noisy_no_ec',marker='d',color='r',s=50,zorder=2,lw=2)
plt.scatter(x1,dif1,label='noisy_w_ec',marker='*',color='b',s=50,zorder=2,lw=2)
#plt.axis()
plt.xlabel('H-H Separation',fontsize=12)
plt.ylabel('Energy, H',fontsize=12)
#plt.title('Dissociation Curve of Linear H$_3$')
plt.legend(loc=4)
plt.show()
