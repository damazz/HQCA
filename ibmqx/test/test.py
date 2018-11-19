import numpy as np
import sys
import test_rdm_tomo as t
sys.path+= ['/home/scott/Documents/research/3_vqa']
import hqca.tools.RDMFunctions as rdmf
from random import randint
np.set_printoptions(precision=4,suppress=True)
def r():
    return randint(0,90)
#parameters = [r(),r(),r(),r(),r(),r()]
parameters = [10,15,20,30,45,5]
print('Parameters: {}'.format(parameters))
backend = 'local_qasm_simulator'
test3,circ3 = t.tomography(parameters,backend='local_unitary_simulator',rdm1=False,rdm2=False,unitary=True)
ON,unit = t.counts_to_1rdm(test3[0],test3[1],unitary=True)
test2, circ2 = t.tomography(parameters,backend,shots=1024,rdm1=False,rdm2=True)
test1, circ1 = t.tomography(parameters,backend,shots=1024,rdm1=True,rdm2=False)
on, compare = t.counts_to_1rdm(test1.get_counts(circ1[0]),test1.get_counts(circ1[1]))
qb_par = {0:1,1:-1,2:1}
rdm2 = t.assemble_2rdm(test2,circ2,qb_par)
rdm1 = rdmf.check_2rdm(rdm2)
rdm1 = np.real(rdm1)
trdm2 = rdmf.trace_2rdm(rdm2)
print('1-RDM from the 2-RDM:')
print(rdm1)
print('1-RDM from direct measurement:')
print(compare)
print(on)
print('1-RDM from Unitary Transformation:')
print(unit)
print(ON)
print('Trace of the 2-RDM: {}'.format(trdm2))
print('Error from direct and unit: {}'.format(np.linalg.norm(compare-unit)))
print('Error from 2-rdm and unit: {}'.format(np.linalg.norm(rdm1-unit)))

