import numpy as np
import pickle
import subprocess
import traceback


def switch(mat):
    size = len(mat.shape)
    L = mat.shape[0]
    if size==2:
        mat = np.reshape(
                mat,
                (
                    int(np.sqrt(L)),
                    int(np.sqrt(L)),
                    int(np.sqrt(L)),
                    int(np.sqrt(L))
                    )
                )
    elif size==4:
        mat = np.reshape(
                mat,
                (
                    L**2,
                    L**2
                    )
                )
    return mat

def contract(mat):
    size = len(mat.shape)
    L = mat.shape[0]
    if size==4:
        mat = np.reshape(
                mat,
                (
                    L**2,
                    L**2
                    )
                )
    return mat

def expand(mat):
    size = len(mat.shape)
    L = mat.shape[0]
    if size==2:
        mat = np.reshape(
                mat,
                (
                    int(np.sqrt(L)),
                    int(np.sqrt(L)),
                    int(np.sqrt(L)),
                    int(np.sqrt(L))
                    )
                )
    return mat




def UnitaryToCounts(Unitary,precision=9):
    '''
    Takes a unitary matrix, and gives a counts representation.
    '''
    N = Unitary.shape[0]
    Nq = int(np.log2(N))
    wf = np.asmatrix(Unitary[0,:]).T
    print(wf)
    prec = 10**(precision)
    wf = (np.real(np.round(np.square(wf)*prec)))
    wf = [int(wf[i,0]) for i in range(0,N)]
    unit = ['{:0{}b}'.format(i,Nq) for i in range(0,N)]
    data = dict(zip(unit,wf))
    print(unit,data)
    return data

def get_counts(filename):
    with open(filename,'rb') as fp:
        data = pickle.load(fp)
    names = []
    for i in range(0,len(data[0]['data'][0]['qobj']['circuits'])):
        names.append(data[0]['data'][0]['qobj']['circuits'][i]['name'])
    counts  = []
    for i in range(0,len(data)):
        counts.append([])
        for j in range(0,len(names)):
            counts[i].append(data[i]['total-counts-{}'.format(names[j])])
    return counts

def get_ideal(filename):
    with open(filename,'rb') as fp:
        data = pickle.load(fp)
    names = []
    for i in range(0,len(data[0]['data'][0]['qobj']['circuits'])):
        names.append(data[0]['data'][0]['qobj']['circuits'][i]['name'])
    counts  = []
    for i in range(0,len(data)):
        counts.append([])
        for j in range(0,len(names)):
            counts[i].append(data[i]['total-counts-{}'.format(names[j])])
    return counts


def get_trace(Nq,active):
    qb_list = [i for i in range(0,Nq)]
    trace_list = []
    for i in qb_list:
        if i in active:
            pass
        else:
            trace_list.append(i)
    return trace_list


def construct_rdm(diag,rot):
    Nq = len(diag)
    qb_sign = {}
    ind = 0
    for item in reversed(range(0,Nq)):
        qb_sign[item]=(-1)**ind
        ind+=1
    Ns = 2*Nq-1
    rdm = np.zeros((Nq*2,Nq*2))
    for i in range(0,Nq):
        rdm[Ns-i,i] = (rot[i]-0.5)*qb_sign[i]
        rdm[i,i] = diag[i]
        rdm[i,Ns-i] = rdm[Ns-i,i]
        rdm[Ns-i,Ns-i] = 1 - rdm[i,i]
    evalues,evector = np.linalg.eig(rdm)
    return rdm,evalues,evector

def rdm(data,unitary=False):
    #takes in a set of counts data in the dictionary form, decodes it, and 
    # obtains the RDM element 
    if unitary==True:
        wf = np.matrix([[1],[0],[0],[0],[0],[0],[0],[0]])
        wf = data*wf
        wf = (np.real(np.round(np.square(wf)*1000000)))
        wf = [int(wf[i,0]) for i in range(0,8)]
        unit = ['000','001','010','011','100','101','110','111']
        data = dict(zip(unit,wf))
    else:
        unit = list(data.keys())
    r = np.zeros(len(unit[0]))
    total_count = 0
    for qubit, count in data.items():
        total_count += count
        n_qb = len(qubit)
        for i in range(0,n_qb):
            if qubit[n_qb-1-i]=='0':
                r[i]+= count
    r = np.multiply(r,total_count**-1) 
    return r

def filt(qb_counts,trace=[]):
    Nqb = len(list(qb_counts.keys())[0])
    Nf = Nqb-len(trace)
    qb_list = ['{:0{}b}'.format(i,Nf) for i in range(0,2**Nf)]
    new_data = {}
    for k,v in qb_counts.items():
        for i in trace:
            i = Nqb-1-i
            if i==(Nqb-1):
                k = k[0:i]
            else:
                k = k[0:i]+k[i+1:]
        if k in qb_list:
            try:
                new_data[k] = new_data[k] + v
            except KeyError:
                new_data[k]=v
        else:
            continue
    hold_sum = 0
    for k,v in new_data.items():
        hold_sum += v
    #print(qb_counts,new_data,hold_sum)
    return new_data

def counts_to_1rdm(
        main,
        err,
        algorithm='ry2p',
        order='default',
        use_err=False,
        **kwargs):
    #try:
    #    trace = get_trace(
    #            algorithm_tomography[algorithm]['Nq'],
    #            algorithm_tomography[algorithm]['qb_to_orb']
    #            )
    #    Nq = algorithm_tomography[algorithm]['Nq']
    #except Exception:
    #    traceback.print_exc()
    if not use_err:
        err_elements = [0.5 for i in range(0,Nq)]
    else:
        err_elements = rdm(filt(err,trace))
    #print(rdm(filt(main,trace)),err_elements)
    ONrdm, ON, ONvec = construct_rdm(
        rdm(
            filt(main,trace)
            ),
        err_elements
        )
    return ON,ONrdm

# maps are from the GPC basis to the chemistry basis
# additionally, there is a chem to spatial basis, which is nearlly trivial
# note, the chemistry basis is ordered... aaabbb, or (alp)(alp)(alp)(bet)(bet)(bet)

map_zeta = {
    0:0, 1:1, 3:2,
    2:3, 4:4, 5:5}
# contaminated 
map_lambda = {
    0:0, 1:1, 3:2,
    2:3, 4:5, 5:4}
# singlet
map_kappa = {
    0:1, 1:0, 3:2,
    2:3, 4:4, 5:5}
# singlet
map_iota = {
    0:1, 1:0, 3:2,
    2:3, 4:5, 5:4}
# contaminated



map_spatial = {
    0:0, 1:1, 2:2,
    3:0, 4:1, 5:2}

def map_wf(wf,mapping):
    new_wf = {}
    for det, val in wf.items():
        new_det = ''
        for i in range(0,len(det)):
            new_det += '0'
        for i in range(0,len(det)):
            ind = mapping[i]
            new_det = new_det[:ind]+det[i]+new_det[ind+1:]
        new_wf[new_det]=val
    return new_wf


def extend_wf(
        wf,
        Norb_tot,
        Nels_tot,
        alpha,
        beta
        ):
    '''Turns active space into full wavefunction'''
    new_wf = {}
    Nso = Norb_tot*2
    for k,v in wf.items():
        new = '0'*Nso
        for i in alpha['inactive']:
            new=new[:i]+'1'+new[i+1:]
        for i in beta['inactive']:
            new=new[:i]+'1'+new[i+1:]
        ind =0
        for i in k:
            if ind<len(alpha['active']):
                if i=='1':
                    z = alpha['active'][ind]
                    new=new[:z]+'1'+new[z+1:]
                else:
                    pass
                ind+=1
            else:
                if i=='1':
                    z = beta['active'][ind-len(beta['active'])]
                    new=new[:z]+'1'+new[z+1:]
                else:
                    pass
                ind+=1 
        new_wf[new]=v
    return new_wf



def get_mapping(mapping):
    if mapping=='zeta':
        return map_zeta
    elif mapping=='lambda':
        return map_lambda
    elif mapping=='kappa':
        return map_kappa
    elif mapping=='iota':
        return map_iota
    else:
        print('Some sort of error in mapping.')
        return None

from math import floor, log10
def round_to_1(x):
    return round(1/x, -int(floor(log10(abs(1/x)))))

def increase(criteria,num_shots,tolerance='default'):
    if tolerance=='default':
        tolerance=round_to_1(num_shots)
    if criteria<= tolerance:
        if num_shots<(8192):
            num_shots*= 2
    return num_shots


def get_reading_material(startover=False):
    '''
    Silly function to get reading material and print it out while waiting for
    the IBM machine to run.

    '''
    loc = '/home/scott/Documents/research/3_vqa/hqca/doc/reading.txt'
    ind = '/home/scott/Documents/research/3_vqa/hqca/doc/track.txt'
    try:
        with open(ind,'r') as fp:
            for line in fp:
                skip_to = int(line)
    except Exception:
        skip_to = 38
    with open(loc,'r') as fp:
        index = 0 
        verses = 0 
        for line in fp:
            if index<skip_to:
                index+=1 
                continue
            else:
                pass
            if verses<11:
                pass
            else:
                with open(ind,'w') as fi:
                    if startover:
                        fi.write(str(38))
                    else:
                        fi.write(str(index))
                break
            if line[0]=='\n':
                verses +=1 
                print('')
            else:
                print(line[:-1])
            index+=1 
            continue

