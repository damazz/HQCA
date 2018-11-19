import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../gpc/')
import simul.run as rn
import numpy.linalg as LA
import gpcf.rdm as rdm
# load data

with open('real.csv','r') as fp:
    data_main = {}
    i = 0
    tally = 0
    for line in fp.readlines():
        if i==0:
            i+=1 
            continue
        line = line.split(',')
        data_main[line[0][1:-1]]=int(line[1][:-1])
        i+=1 
        tally += int(line[1][:-1])
with open('sim.csv','r') as fp:
    data_lsim = {}
    i = 0
    tally = 0
    for line in fp.readlines():
        if i==0:
            i+=1 
            continue
        line = line.split(',')
        data_lsim[line[0][1:-1]]=int(line[1][:-1])
        i+=1 
        tally += int(line[1][:-1])

# get ideal
wf = rn.single_run_c3(43,3,39,[0,2,0,1,2,1])
nrdm = rn.construct_rdm(wf)
noc,nov = LA.eig(nrdm)
noc.sort()
# first, generate stochastically generated data
data_main = rdm.filt(data_main,trace=[0,1])
data_lsim = rdm.filt(data_lsim,trace=[0,1])

main_hold = []
for key, val in data_main.items():
    q = 0
    while q<val:
        main_hold.append(key)
        q+=1 
        
lsim_hold = []
for key, val in data_lsim.items():
    q = 0
    while q<val:
        lsim_hold.append(key)
        q+=1 
    

main_list = np.random.choice(main_hold,len(main_hold),replace=False)
lsim_list = np.random.choice(lsim_hold,len(lsim_hold),replace=False)

new_main = {
    '000':0,
    '001':0,
    '010':0,
    '011':0,
    '100':0,
    '101':0,
    '110':0,
    '111':0,
    }
new_lsim = {
    '000':0,
    '001':0,
    '010':0,
    '011':0,
    '100':0,
    '101':0,
    '110':0,
    '111':0,
    }
Y_main = np.zeros(8192)
Y_lsim = np.zeros(8192) 
unit = 8192
#X = np.linspace(0,1,8192)
X = np.arange(1,8193)
for i in range(0,unit):
    mresult = main_list[i]
    sresult = lsim_list[i]
    new_main[mresult] = new_main[mresult]+1
    new_lsim[sresult] = new_lsim[sresult]+1

    mrdm,mnoc,mnor = rdm.construct_rdm(rdm.rdm(new_main),[0.5,0.5,0.5])
    srdm,snoc,snor = rdm.construct_rdm(rdm.rdm(new_lsim),[0.5,0.5,0.5])
    mnoc.sort()
    snoc.sort()
    Y_main[i] = np.log10(np.sqrt(np.sum(np.square(mnoc[3:]-noc[3:]))))
    Y_lsim[i] = np.log10(np.sqrt(np.sum(np.square(snoc[3:]-noc[3:]))))
    if i%100==0:
        print(i/unit)
#rdm, roc, rvec = rdm.construct_rdm(rdm.rdm(new_data),[0.5,0.5,0.5])
    
g2 = 1-0.00189
g3 = 1-0.00197
g4 = 1-0.00103
r2 = 1-0.0310
r3 = 1-0.0250
r4 = 1-0.0440
c32 = 1-0.0365
c34 = 1-0.0366
c42 = 1-0.0334
times = 1
gate_error = (g3**times)*c32*(g3**times)*c34*(g4**times)*c42
readout = r2*r3*r4
print('Gate Error: {}'.format(gate_error))
print('Readout Error: {}'.format(readout))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(X,Y_main[0:unit],c='r',label='QC')
ax1.plot(X,Y_lsim[0:unit],c='g',label='sQS')
ax1.legend()
ax1.set_xlim(0,unit)
ax1.set_xlabel('# of shots')
ax1.set_ylabel('log$_{10}$ Distance from Ideal Point')
ax1.set_title('Comparison between Quantum Computer and Quantum Simulator')
'''
def autocorr(x):
    result = np.correlate(x,x,mode='full')
    return result[len(result)//2:]
y_lsim = autocorr(np.exp(Y_lsim[0:unit]))
y_main = autocorr(np.exp(Y_main[0:unit]))

ax2 = fig.add_subplot(212)
ax2.plot(X,y_main,c='r')
ax2.plot(X,y_lsim,c='g')
ax2.set_xlim=(0,1)
ax2.legend()
'''
plt.show()







# then, generate 






