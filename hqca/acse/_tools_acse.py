import numpy as np
from copy import deepcopy as copy
from hqca.operators import *
from hqca.tools import *

class Log:
    def __init__(self,
            energy=False,
            norm=False,
            A=False,
            variance=False,
            rdm=False,
            psi=False,
            cnot = False,
            gamma = False,
            opt = False,
            counts= False,
            ):
        self.norm = []
        if energy:
            self.E = []
        if norm:
            self.norm = []
        if variance:
            self.sigma = []
        if rdm:
            self.rdm = []
        if A:
            self.A = []
        if cnot:
            self.cx = []
        if gamma:
            self.gamma = []
        if opt: #log of logs
            self.opts = []
        if counts:
            self.counts = []

def truncation(
        vector,rel_thresh,min_thresh,
        method='abs',
        include=False,
        gradient=None,
        hessian=None,
        previous=False,
        keys=None,
        mapping=None,
        E0=0):
    '''
    contains truncation schemes for the different vectors

    truncation is constrained so that we maintain a good search direction
    '''
    dim = vector.shape[0]
    #print('vector')
    #b = np.nonzero(vector)
    #for j,l in zip(b[0],b[1]):
    #    if abs(vector[j,l])>1e-8:
    #        print(j,vector[j,l])
    #print('grad')
    #b = np.nonzero(gradient)
    #for j,l in zip(b[0],b[1]):
    #    if abs(gradient[j,l])>1e-8:
    #        print(j,gradient[j,l])
    k_added = []
    #try:
    #    for k,i in enumerate(previous):
    #        print('step {}'.format(k))
    #        b = np.nonzero(i)
    #        for j,l in zip(b[0],b[1]):
    #            if abs(i[j,l])>1e-8:
    #                print(j,i[j,l])
    #    print('-- -- --')
    #except Exception:
    #    pass
    if include:
        for i in range(dim):
            for k in range(len(previous)):
                if abs(previous[k][i,0])>=min_thresh:
                    # then, yes we used this term before!
                    k_added.append(i)
    #
    deltas = np.multiply(vector,gradient)

    if method=='abs':
        crit = np.argsort(np.abs(vector.flatten())).tolist()
        if len(crit)==1:
            crit = crit[0]
        max_element = abs(max(vector.max(), vector.min(), key=abs))
        thresh = max(max_element * rel_thresh, min_thresh)
    elif method=='delta':
        crit = np.argsort(np.asarray(deltas).flatten()).tolist()
        max_element = deltas.min()
        thresh = max_element * rel_thresh
    else:
        raise OptimizationError('No criteria selected')
    count = 0
    ind = 0
    descent = np.sum(deltas)
    done = []
    temp = copy(vector)
    check = copy(crit)
    while len(check)>0:
        # 
        # check nonzero
        # check if should be removed
        # check if removing would result in non-descent direction
        # then 0 

        # for pairs 

        target = check[0] # cycle through elements
        pair_inds = keys[target]
        inds = pair_inds[2:]+pair_inds[:2]
        pair = mapping[' '.join([str(k) for k in inds])]

        if target==pair:
            check.remove(target)
            continue

        if include and target in k_added:
            check.remove(target)
            check.remove(pair)
            continue
        remove = False
        if method=='abs':
            val = abs(vector[target])
            if val<thresh*0.99999:
                remove = True
    
        if method in ['del','delta']:
            val = deltas[target]
            if val>thresh*0.99999:
                remove = True
        
        en = deltas[target]+deltas[target]
        descent = np.dot(temp.T,gradient) - en 
        if descent>=0:
            remove=False

        if remove:
            temp[target] =0
            temp[pair]   =0
        check.remove(target)
        check.remove(pair)

    nz = np.nonzero(temp)
    #print('vec')
    #for i,j in zip(nz[0],nz[1]):
    #    if abs(temp[i,j])>min_thresh:
    #        if not i in k_added:
    #            print(i,temp[i,j],deltas[i],keys[i])
    #print('k-added')
    #for i,j in zip(nz[0],nz[1]):
    #    if abs(temp[i,j])>min_thresh:
    #        if i in k_added:
    #            print(i,temp[i,j],deltas[i],keys[i])

    return temp

def vector_to_operator(vector,fermi=True,user=False,keys=[],N=10,S_min=1e-10):
    '''
    translates a compact vector format to an operator using a list of keys
    that associates values of the vector to 2-RDM operators
    '''
    if user:
        op = Operator()
        for k,v in zip(keys,vector):
            if abs(v[0])>=1e-12:
                op+= PauliString(k,-1j*v[0])
    else:
        op = Operator()
        for i in range(len(keys)):
            # convert indices
            if len(vector.shape)==2:
                if abs(vector[i,0])<=S_min:
                    continue
                val = vector[i,0]
            else:
                val = vector[i]

            if fermi:
                # note, for is for the double counting in the RDM to operator
                # because elements are all are counted in the operator -> RDM, but
                # since we are using a compact representation, we can muiltiply this by 4
                # [i,k,j,l] [k,i,j,l] [k,i,l,j], [i,k,l,j]
                # we fix the indices so that the sign is correct
                inds = copy(keys[i])
                il = inds[:2]
                ir = inds[2:][::-1] #
                new = il+ir
                op += FermiString(
                    4*val,
                    indices=new,
                    ops='++--',
                    N=N,
                )
                #op -= FermiString(
                #    4*val,
                #    indices=(il+ir)[::-1],
                #    ops='++--',
                #    N=N,
                #)
            else:
                inds = copy(keys[i])
                il = inds[:2]
                ir = inds[2:][::-1] #
                op += QubitString(
                    4*val,
                    indices=il+ir,
                    ops='++--',
                    N=N,
                )
    return op

