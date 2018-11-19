#quantum simulator - homemade program, not provided by IBM
import numpy as np
import simul
from simul import function as fx
from simul import run
import qiskit
from qiskit import QuantumProgram, QuantumCircuit
from qiskit.tools.visualization import plot_histogram,plot_state
from qiskit.tools.qi.qi import state_fidelity,concurrence,purity,outer
from qiskit.tools.qcvv import tomography as tomo

np.set_printoptions(precision=3,linewidth=200)
#
# first, need to test the functionality
#
qp = QuantumProgram()
qreg_1 = qp.create_quantum_register('qr1',3)
creg_1 = qp.create_classical_register('cr1',3)
circ_1 = qp.create_circuit('circ_1',[qreg_1],[creg_1])

qreg_2 = qp.create_quantum_register('qr2',3)
creg_2 = qp.create_classical_register('cr2',3)
circ_2 = qp.create_circuit('circ_2',[qreg_2],[creg_2])

circ_1.ry(2*(np.pi/6),qreg_1[0])
circ_1.cx(qreg_1[0],qreg_1[2])
circ_1.ry(2*(np.pi/6),qreg_1[0])
circ_1.cx(qreg_1[0],qreg_1[1])
circ_1.ry(2*(np.pi/6),qreg_1[2])
circ_1.cx(qreg_1[2],qreg_1[1])

circ_2.ry(2*(np.pi/6),qreg_2[0])
circ_2.cx(qreg_2[0],qreg_2[2])
circ_2.ry(2*(np.pi/6),qreg_2[0])
circ_2.cx(qreg_2[0],qreg_2[1])
circ_2.ry(2*(np.pi/6),qreg_2[2])
circ_2.cx(qreg_2[2],qreg_2[1])

#circ_2.measure(qreg_2[0],creg_2[0])
#circ_2.measure(qreg_2[1],creg_2[1])
#circ_2.measure(qreg_2[2],creg_2[2])

results = qp.execute('circ_2',backend='local_qasm_simulator',shots=1)
tf = results.get_data('circ_2')['quantum_state']
print(tf)
results = qp.execute('circ_1',backend='local_unitary_simulator')
tf = results.get_data('circ_1')['unitary']
wf = tf[:,0]
print(wf)
#results = qp.execute('circ_1',backend='local_qasm_simulator')

#tf = results.get_data('circ_1')['counts']
#print(tf)

#tf = np.asmatrix(tf)
#wf = np.matrix([[0],[0],[0],[0],[0],[0],[0],[1]])
#wfs = np.array([0,0,0,0,0,0,0,1])
#wf = tf*wf
#wf = (wf.T).tolist()[0]
#print(wf)
tomo_circuits = tomo.build_state_tomography_circuits(qp,'circ_1',[0,1,2],qreg_1,creg_1)


test_tomo = qp.execute(tomo_circuits,shots=1024,backend='local_qasm_simulator')


data = tomo.state_tomography_data(test_tomo,'circ_1',[0,1,2])
print(data)
rho_fit = tomo.fit_tomography_data(data)
print(rho_fit)
F_fit = state_fidelity(rho_fit,wf)
#con = concurrence(rho_fit)
pur = purity(rho_fit)


print('Fidelity =', F_fit)
#print('concurrence =', con)
print('purity =', pur)
plot_state(rho_fit,'paulivec')

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



