import pickle
import numpy as np
import sys
sys.path.append('../../../gpc/')
from gpcf import rdm


def euc(one,two):
    return np.sqrt(np.sum(np.square(one-two)))

with open('e04.dat','rb') as fp:
    data = pickle.load(fp)



sim_occ = []
para = []
for i in ['a','b','c','d','e']:
    dat = np.loadtxt('e04_03_{}.sim'.format(i))
    for sim in dat:
        sim_occ.append(sim.tolist())

sim_par = np.loadtxt('e04.sim')
sim_par = sim_par[:,0:3]
sim_par *=  180/np.pi

qblist = ['00101','00110','00011','00000']
ind = 0 
for item in data:
    test = item['qasms'][0]['qasm']
    new = test.split('\n')
    one = float(new[6].split('(')[1].split(')')[0].split(',')[0])*90/np.pi
    two = float(new[10].split('(')[1].split(')')[0].split(',')[0])*90/np.pi
    thr = float(new[17].split('(')[1].split(')')[0].split(',')[0])*90/np.pi
    counts = item['qasms'][0]['result']['data']['counts']
    filt = {}
    for qb,val in counts.items():
        if qb in qblist:
            filt[qb]=int(val)
    rdme_m = rdm.rdm(counts)
    #print(counts)
    print(filt)
    rdme_f = rdm.rdm(filt)
    occ_m = np.zeros(6)
    occ_f = np.zeros(6)
    for i in range(0,3):
        occ_m[i] = rdme_m[2+i]
        occ_m[5-i] = 1-rdme_m[2+i]
        occ_f[i] = rdme_f[2+i]
        occ_f[5-i]= 1-rdme_f[2+i]
    occ_m.sort()
    occ_f.sort()
    #print(occ_m)
    print('#{}'.format(ind+1))
    print('Parameters: {:.1f},{:.1f},{:.1f}'.format(one,two,thr))
    print('Check Parameters: {:.1f},{:.1f},{:.1f}'.format(sim_par[ind][0],sim_par[ind][1],sim_par[ind][2]))
    print('Difference in raw results: {}'.format(euc(occ_m[3:],sim_occ[ind][3:])))
    print('Difference in fil results: {}'.format(euc(occ_f[3:],sim_occ[ind][3:])))
    #if (ind>100):
    #    print(item['qasms'][0]['qasm'])
    print('')
    ind += 1 
