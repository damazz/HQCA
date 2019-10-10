from hqca.quantum.QuantumFunctions import QuantumStorage
from hqca.tools.Fermi import FermiOperator
import numpy as np
import sys
from hqca.acse.BuildAnsatz import Ansatz

'''
/hqca/acse/FunctionsQACSE.py

Contains functions for performing ACSE calculations, with a focus on generating
the S matrix through time evolution of the Hamiltonian. 
'''

def findSPairsQuantum(Store,QuantStore,verbose=False):
    '''
    need to do following:
        1. prepare the appropriate Hailtonian circuit
        2. implement it
        3. find S from resulting matrix
    '''
    if verbose:
        print('Generating new S pairs with Hamiltonian step.')
    newS = []
    newPsi = Ansatz(Store,QuantStore,propagateTime=True,scalingHam=1.0)
    newPsi.build_tomography(real=False,imag=True) #note we need imaginary as well
    if verbose:
        print('Running circuits...')
    newPsi.run_circuit(verbose=verbose)
    if verbose:
        print('Constructing the RDMs...')
    newPsi.construct_rdm()
    new = np.nonzero(np.imag(newPsi.rdm2.rdm))
    newS = []
    max_val = 0 
    for i,k,j,l in zip(new[0],new[1],new[2],new[3]):
        if abs(np.imag(newPsi.rdm2.rdm)[i,k,j,l])>max_val:
            max_val = abs(np.imag(newPsi.rdm2.rdm)[i,k,j,l])
    print('Max S val: {}'.format(max_val))
    for i,k,j,l in zip(new[0],new[1],new[2],new[3]):
        val = np.imag(newPsi.rdm2.rdm)[i,k,j,l]
        if abs(val)>0.1*max_val:
            c1 =  (i in QuantStore.alpha['active'])
            c2 =  (k in QuantStore.alpha['active'])
            c3 =  (l in QuantStore.alpha['active'])
            c4 =  (j in QuantStore.alpha['active'])
            spin = '{}{}{}{}'.format(
                    c1*'a'+(1-c1)*'b',
                    c2*'a'+(1-c2)*'b',
                    c3*'a'+(1-c3)*'b',
                    c4*'a'+(1-c4)*'b',
                    )
            newEl = FermiOperator(
                    -val,
                    indices=[i,k,l,j],
                    sqOp='++--',
                    spin=spin
                    )
            if len(newS)==0:
                newS.append(newEl)
                if verbose:
                    print('S: {},{},{},{}: {}'.format(i,k,l,j,val))
                    print(newEl.qOp,newEl.qInd,newEl.qSp)
            else:
                add = True
                for o in newS:
                    if o.isSame(newEl) or o.isHermitian(newEl):
                        add = False
                        break
                if add:
                    newS.append(newEl)
                    if verbose:
                        print('S: {},{},{},{}: {}'.format(i,k,l,j,val))
                        print(newEl.qOp,newEl.qInd,newEl.qSp)
    # lets try reversed order
    #hf = []
    #for i in range(QuantStore.Ne_alp):
    #    hf.append(Store.alpha_mo['active'][i])
    #for i in range(QuantStore.Ne_bet):
    #    hf.append(Store.beta_mo['active'][i])

    #def overlap(op,hf):
    #    if op.opType=='de':
    #        if len(set(op.qInd).intersection(set(hf)))==2:
    #            return True
    #    elif op.opType=='ne':
    #        ovlp = len(set(op.qInd).intersection(set(hf)))
    #        if op.num in hf and ovlp==1:
    #            pass
    ## find the naive overlap with the ground state 
    ## note.....ne prrq should have overlap 2 if r in hf
    ##    well......if ovlp 
    ## note.....de should have overlap 2

    ##ovlp = [overlap(op.qInd,hf) for op in newS]
    #hold_type = [(op.opType=='de') for op in newS]
    #new_S_ord = []
    #for i in range(len(hold_type)):
    #    if hold_type[i]:
    #        new_S_ord.append(newS[i])
    #for i in range(len(hold_type)):
    #    if not hold_type[i]:
    #        new_S_ord.append(newS[i])
    #newS = new_S_ord[:]
    return newS

