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

def findSPairsQuantum(Store,QuantStore):
    '''
    need to do following:
        1. prepare the appropriate Hailtonian circuit
        2. implement it
        3. find S from resulting matrix
    '''
    newS = []
    newPsi = Ansatz(Store,QuantStore,propagateTime=True,scalingHam=1.0)
    newPsi.build_tomography(full=True,real=False,imag=True) #note we need imaginary as well
    newPsi.run_circuit()
    newPsi.construct_rdm()
    newPsi.rdm2.switch()
    #print(np.real(newPsi.rdm2.rdm))
    print(np.imag(newPsi.rdm2.rdm))
    newPsi.rdm2.switch()
    new = np.nonzero(np.imag(newPsi.rdm2.rdm))
    newS = []
    for i,k,j,l in zip(new[0],new[1],new[2],new[3]):
        val = np.imag(newPsi.rdm2.rdm)[i,k,j,l]
        if abs(val)>0.01:
            newEl = set([i,k,j,l])
            if not len(newEl)==4:
                continue
            if len(newS)==0:
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
                newOp = FermiOperator(
                        coeff=val,
                        indices=[i,k,l,j],
                        sqOp='++--',
                        spin=spin)
                newS.append(newOp)
            else:
                for i in newS:
                    if i.as_set.difference(newEl):
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
                        newOp = FermiOperator(
                                coeff=val,
                                indices=[i,k,l,j],
                                sqOp='++--',
                                spin=spin)
                        newS.append(newOp)
    return newS

