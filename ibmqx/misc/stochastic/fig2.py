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
keys = {
    0:'000',
    1:'001',
    2:'010',
    3:'011',
    4:'100',
    5:'101',
    6:'110',
    7:'111'
    }
Y_main = np.zeros((8192,8))
Y_lsim = np.zeros((8192,8)) 
unit = 8192
#X = np.linspace(0,1,8192)
X = np.arange(1,8193)
for i in range(0,unit):
    mresult = main_list[i]
    sresult = lsim_list[i]
    new_main[mresult] = new_main[mresult]+1
    new_lsim[sresult] = new_lsim[sresult]+1
    for j in range(0,8):
        Y_lsim[i,j] = new_lsim[keys[j]]*100/(i+1)
    if i%100==0:
        print(i/unit)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(X,Y_lsim[0:unit,0],c='r',label=keys[0])
ax1.plot(X,Y_lsim[0:unit,3],c='g',label=keys[3])
ax1.plot(X,Y_lsim[0:unit,5],c='b',label=keys[5])
ax1.plot(X,Y_lsim[0:unit,6],c='k',label=keys[6])
#ax1.plot(X,Y_lsim[0:unit],c='g',label='sQS')
ax1.legend()
ax1.set_xlim(1,64)
ax1.set_xlabel('# of shots')
ax1.set_ylabel('% of Total Result as Particular Qubit State')
ax1.set_title('Development of QC Histogram w.r.t # of Shots')
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






