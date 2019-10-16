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

def findSPairsQuantum(Store,QuantStore,verbose=False,separate=False):
    '''
    need to do following:
        1. prepare the appropriate Hailtonian circuit
        2. implement it
        3. find S from resulting matrix
    '''
    if separate:
        PsiH = Ansatz(
                Store,QuantStore,
                propagateTime=True,Hamiltonian='split-K')
        PsiV = Ansatz(
                Store,QuantStore,
                propagateTime=True,Hamiltonian='split-V')
        PsiH.build_tomography(real=False,imag=True)
        PsiV.build_tomography(real=False,imag=True)
        PsiH.run_circuit()
        PsiV.run_circuit()
        newH = np.nonzero(np.imag(PsiH.rdm2.rdm))
        newV = np.nonzero(np.imag(PsiV.rdm2.rdm))
        newS = []
        for i,k,j,l in zip(newH[0],newH[1],newH[2],newH[3]):
            if abs(np.imag(PsiH.rdm2.rdm)[i,k,j,l])>max_val:
                max_val = abs(np.imag(newPsiH.rdm2.rdm)[i,k,j,l])
        for i,k,j,l in zip(new[0],new[1],new[2],new[3]):
            val = np.imag(PsiH.rdm2.rdm)[i,k,j,l]
            if abs(val)>0.1*max_val: #and abs(val)>0.01:
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
        for i,k,j,l in zip(newV[0],newV[1],newV[2],newV[3]):
            if abs(np.imag(PsiV.rdm2.rdm)[i,k,j,l])>max_val:
                max_val = abs(np.imag(PsiV.rdm2.rdm)[i,k,j,l])
        for i,k,j,l in zip(new[0],new[1],new[2],new[3]):
            val = np.imag(PsiV.rdm2.rdm)[i,k,j,l]
            if abs(val)>0.1*max_val: #and abs(val)>0.01:
                print('Si: {:.6f}:{}{}{}{}'.format(val,i,k,j,l))
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
                        if o.isSame(newEl):
                            add = False
                            o.qCo+= newEl.qCo
                            o.c  += newEl.c
                            break
                        elif o.isHermitian(newEl):
                            add = False
                            break
                    if add:
                        newS.append(newEl)
                        if verbose:
                            print('S: {},{},{},{}: {}'.format(i,k,l,j,val))
                            print(newEl.qOp,newEl.qInd,newEl.qSp)
    elif not separate:
        if verbose:
            print('Generating new S pairs with Hamiltonian step.')
        newS = []
        newPsi = Ansatz(Store,QuantStore,propagateTime=True,scalingHam=2.0)
        newPsi.build_tomography(real=False,imag=True) 
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
        #print('Max S val: {}'.format(max_val))
        for i,k,j,l in zip(new[0],new[1],new[2],new[3]):
            val = np.imag(newPsi.rdm2.rdm)[i,k,j,l]
            if abs(val)>0.1*max_val: #and abs(val)>0.01:
                #print('Si: {:.6f}:{}{}{}{}'.format(val,i,k,j,l))
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
    hold_type = [(op.opType=='de') for op in newS]
    new_S_ord = []
    for i in range(len(hold_type)):
        if hold_type[i]:
            new_S_ord.append(newS[i])
    for i in range(len(hold_type)):
        if not hold_type[i]:
            new_S_ord.append(newS[i])
    newS = new_S_ord[:]
    return newS

