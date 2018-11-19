#quantum simulator - homemade program, not provided by IBM

import sys
sys.path.append('/home/scott/Documents/research/3_vqa/hqca/ibmqx')

import numpy as np
import simul
from simul import function as fx
from simul import run
import qiskit
from qiskit import QuantumProgram, QuantumCircuit
np.set_printoptions(precision=3,linewidth=200)
#
# first, need to test the functionality
#
qp = QuantumProgram()
qreg_1 = qp.create_quantum_register('qr1',3)
creg_1 = qp.create_classical_register('cr1',3)
circ_1 = qp.create_circuit('circ_1',[qreg_1],[creg_1])

circ_1.ry(2*(np.pi/6),qreg_1[0])
circ_1.cx(qreg_1[0],qreg_1[2])
circ_1.ry(2*(np.pi/6),qreg_1[0])
circ_1.cx(qreg_1[0],qreg_1[1])
circ_1.ry(2*(np.pi/6),qreg_1[2])
circ_1.cx(qreg_1[2],qreg_1[1])

circ_1.measure(qreg_1[0],creg_1[0])
circ_1.measure(qreg_1[1],creg_1[1])
circ_1.measure(qreg_1[2],creg_1[2])

results = qp.execute('circ_1',backend='local_unitary_simulator')
#results = qp.execute('circ_1',backend='local_qasm_simulator')

tf = results.get_data('circ_1')['unitary']
#tf = results.get_data('circ_1')['counts']
#print(tf)

tf = np.asmatrix(tf)
wf = np.matrix([[0],[0],[0],[0],[0],[0],[0],[1]])
wfs = np.array([1,0,0,0,0,0,0,0])
wf = tf*wf
print(wf)

tf = []
tf.append(fx.m_i(0,fx.rot(30)))
tf.append(fx.m_ij(0,2,fx.CNOT)) #CLASS 1	
tf.append(fx.m_i(0,fx.rot(30)))
tf.append(fx.m_ij(0,1,fx.CNOT)) #CLASS 2	
tf.append(fx.m_i(2,fx.rot(30)))
tf.append(fx.m_ij(2,1,fx.CNOT)) #CLASS 3	
wfs = fx.mml(tf,wfs)
print(np.asmatrix(wfs).T)

def check_wf(wf_ses,wf_ibm):
    print(wf_ses[0]-wf_ibm[7]) 
    print(wf_ses[1]-wf_ibm[3]) 
    print(wf_ses[2]-wf_ibm[5]) 
    print(wf_ses[3]-wf_ibm[1]) 
    print(wf_ses[4]-wf_ibm[6]) 
    print(wf_ses[5]-wf_ibm[2]) 
    print(wf_ses[6]-wf_ibm[4]) 
    print(wf_ses[7]-wf_ibm[0]) 
    return


check_wf(wfs,wf)
#simul.run.debug_run(30,30,30,[0,2,0,1,2,1],1e-4)
# ibm ses  map
# 210 012  --- 
# --- --- -----
# 000 111 0<->7
# 001 110 1<->3
# 010 101 2<->5
# 011 100 3<->1
# 100 011 4<->6
# 101 010 5<->2
# 110 001 6<->4
# 111 000 7<->0



