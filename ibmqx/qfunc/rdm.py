import numpy as np
from numpy import linalg as LA

def filt(qb_counts,trace=[],qb_dict=['000','001','010','011','100','101','110','111']):
    n_qb = len(list(qb_counts.keys())[0])
    new_data = {}
    for k,v in qb_counts.items():
        for i in trace:
            i = 4-i
            if i==4:
                k = k[0:i]
            else:
                k = k[0:i]+k[i+1:]
        if k in qb_dict:
            pass
        else:
            continue
        #print(k,v)
        try:
            new_data[k] = new_data[k] + v
        except KeyError:
            new_data[k]=v

    sum = 0
    for k,v in new_data.items():
        sum += v
    #print(qb_counts,new_data,sum)
    #print(new_data)
    return new_data

def U_to_counts(U_mat,precision=9):
    '''
    Takes a unitary matrix, and gives a counts representation.
    '''
    N = U_mat.shape[0]
    Nq = int(np.log2(N))
    wf = np.asmatrix(U_mat[0,:]).T
    print(wf)
    prec = 10**(precision)
    wf = (np.real(np.round(np.square(wf)*prec)))
    wf = [int(wf[i,0]) for i in range(0,N)]
    unit = ['{:0{}b}'.format(i,Nq) for i in range(0,N)]
    data = dict(zip(unit,wf))
    print(unit,data)
    return data


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
                r[n_qb-1-i]+= count
    r = np.multiply(r,total_count**-1) 
    return r

    
    

def rdme(data,unitary=False,qubits=[0,1,2],filt=[]):
    if unitary==True:
        wf = np.matrix([[1],[0],[0],[0],[0],[0],[0],[0]])
        wf = data*wf
        wf = (np.real(np.round(np.square(wf)*1000000)))
    return None 

def construct_rdm(diag,rot):
    rdm = np.zeros((6,6))
    for i in range(0,len(rot)):
        rdm[5-i,i] = 0.5 - rot[i]
        rdm[i,i] = diag[i]
        if i==1:
            rdm[5-1,i]*= -1
        rdm[i,5-i] = rdm[5-i,i]
        rdm[5-i,5-i] = 1 - rdm[i,i]
    evalues,evector = LA.eig(rdm)
    return rdm,evalues,evector
 
    
