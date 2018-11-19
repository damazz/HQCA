import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pickle
from qiskit import QuantumProgram
import qiskit.tools.qi.qi as qi
import qiskit
sys.path.append('../../../gpc/')
from gpcf import rdm
from gpcf import gpc
from simul import run
import numpy.linalg as LA

with open(sys.argv[1],'rb') as fp:
    data = pickle.load(fp)

discard = False
diag = False

n = len(data)

sort_data = []

class deconstruct():
    def __init__(self,ibm_result):
        #check if done first
        self.use = (ibm_result['status']=='COMPLETED')
        if self.use:
            self.main_qasm = ibm_result['qasms'][0]['qasm']
            self.err_qasm = ibm_result['qasms'][1]['qasm']
            self.main_counts  = ibm_result['qasms'][0]['result']['data']['counts']
            self.err_counts  = ibm_result['qasms'][1]['result']['data']['counts']
            self.parameters = []
            for line in self.main_qasm.split('\n'):
                if line[0:2]=='u3':
                    nline = line.split('(')[1]
                    nline = nline.split(')')[0]
                    nline = nline.split(',')[0]
                    self.parameters.append(float(nline))


def combine_dict(one,two):
    for key,val in two.items():
        try:
            one[key] = int(one[key]) + int(val)
        except:
            one[key] = int(val)
    return one


new_data = []
point = 0
for i in range(0,n//2):
#for i in range(5,6):
    a = deconstruct(data[2*i])
    b = deconstruct(data[2*i+1])
    new_data.append({})
    new_data[point]['main-counts']= combine_dict(a.main_counts,b.main_counts)
    new_data[point]['err-counts'] = combine_dict(a.err_counts,b.err_counts)
    new_data[point]['parameters'] = a.parameters
    #for j in a.parameters:
    #    print(j*90/np.pi)
    new_data[point]['main-qasm'] = a.main_qasm
    new_data[point]['err-qasm'] = a.err_qasm

    exp_main = rdm.rdm(rdm.filt(new_data[point]['main-counts'],trace=[0,1]))
    exp_err = rdm.rdm(rdm.filt(new_data[point]['err-counts'],trace=[0,1]))
    print(exp_err)
         
    exp_rdm,exp_ON,exp_vec = rdm.construct_rdm(exp_main,exp_err)
    exp_ON.sort()
    new_data[point]['exp-ON'] = exp_ON.tolist()
    print('Experimental: {}'.format(exp_ON.tolist()))
    ##
    if diag:
        qb_key = ['000','110','011','101']
        diag_main = rdm.rdm(rdm.filt(new_data[point]['main-counts'],trace=[0,1],qb_dict=qb_key))
        print(diag_main)
        diag_rdm,diag_ON,diag_vec = rdm.construct_rdm(diag_main,[0.5,0.5,0.5])
        diag_ON.sort()
        new_data[point]['diag-ON'] = diag_ON.tolist()
        print('Filtered: {}'.format(diag_ON.tolist()))

    ##
    qp = QuantumProgram()
    qp.load_qasm_text(a.main_qasm,'main')
    qp.load_qasm_text(b.err_qasm,'err')
    results_u = qp.execute(['main','err'],backend='local_unitary_simulator')
    unit_main = results_u.get_data('main')['unitary']
    unit_err = results_u.get_data('err')['unitary']
    unit_main=qi.partial_trace(unit_main,[0,1])/4
    unit_err=qi.partial_trace(unit_err,[0,1])/4
    
    rdme_main = rdm.rdm(unit_main,unitary=True)
    rdme_err  = rdm.rdm(unit_err,unitary=True)
    unit_rdm,unit_ON,unit_vec = rdm.construct_rdm(rdme_main,rdme_err)
    unit_ON.sort()
    print('Unitary: {}'.format(unit_ON))
    new_data[point]['ideal'] = unit_ON.tolist()
    simul = False
    if simul:
        p = np.array(a.parameters)*(90/np.pi)
        try:
            sim = run.single_full_run_c3(p[1],p[2],p[3],p[0],p[4],p[5],[0,1,1,2,2,0])
            #sim = run.single_full_run_c3(p[1],p[2],p[3],p[0],p[4],p[5],[1,0,1,2,2,0])
        except:
            sim = run.single_full_run_c3(p[0],0,p[1],0,p[2],0,[0,2,0,1,2,1])
        simON, simVEC = LA.eig(run.construct_rdm(sim))
        simON.sort()
        print('Simulator: {}'.format(simON))
    point+=1   
    try:
        dist_filt= np.sqrt(np.sum(np.square(unit_ON[3:]-diag_ON[3:])))
        dist_exp= np.sqrt(np.sum(np.square(unit_ON[3:]-exp_ON[3:])))
        print('Distances, exp and filt: {}, {}'.format(dist_exp,dist_filt))
    except:    
        dist_exp= np.sqrt(np.sum(np.square(unit_ON[3:]-exp_ON[3:])))
        print('Distances, exp: {}'.format(dist_exp))
    print(point,'\n')

savefile = sys.argv[1].split('.')[0]+'.out'



# i.e., diagonal wavefunction form
if discard:
    try: 
        use = (np.loadtxt(sys.argv[1].split('.')[0]+'.use')).tolist()
        print(use)
        ind = len(use)-1
        for i in reversed(use):
            if not i:
                print('deleting: {}'.format(ind))
                new_data.pop(ind)
                ind-=1
            else:
                ind-=1
    except Exception as e:
        print(e)
print(len(new_data))
    
with open(savefile,'wb') as fp:
    pickle.dump(new_data,fp,0)
            
