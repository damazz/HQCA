from qiskit import QuantumProgram, QuantumCircuit
from qiskit.tools.visualization import plot_histogram,plot_state
from numpy import pi 
import gpc_var.rdm
import numpy as np
#from qiskit import Qconfig
from qiskit.tools.qi.qi import state_fidelity,concurrence,purity,outer
qp = QuantumProgram()
backends = 'local_qasm_simulator'
np.set_printoptions(linewidth=200,suppress=True,precision=4)
# initiate program
qm = qp.create_quantum_register("main_qr",3)
qe = qp.create_quantum_register("err_qr",3)
cm = qp.create_classical_register("main_cr",3)
ce = qp.create_classical_register("err_cr",3)

main = qp.create_circuit('main',[qm],[cm])
err  = qp.create_circuit('err',[qe],[ce])

theta1 = pi/3
theta2 = pi/6
theta3 = pi/12
thetao = pi/4
thetao2 = pi/4


main.ry(2*theta1, qm[0])
main.cx(qm[0],qm[2])
main.ry(2*theta2, qm[0])
main.cx(qm[0],qm[1])
main.ry(2*theta3,qm[2])
main.cx(qm[2],qm[1])
#131232


main.measure(qm[0],cm[0])
main.measure(qm[1],cm[1])
main.measure(qm[2],cm[2])
    

err.ry(2*theta1, qe[0])
err.cx(qe[0],qe[2])
err.ry(2*theta2, qe[0])
err.cx(qe[0],qe[1])
err.ry(2*theta3,qe[2])
err.cx(qe[2],qe[1])
err.ry(2*thetao,qe[0])
err.ry(2*thetao2,qe[1])
err.ry(2*thetao,qe[2])
err.measure(qe[0],ce[0])
err.measure(qe[1],ce[1])
err.measure(qe[2],ce[2])

circuits = ['main','err']
#UnitarySimulator(main).run()

if backends=='local_unitary_simulator':
    num = 1
elif backends=='local_qasm_simulator':
    num = 2048
result = qp.execute(circuits,shots=num,backend=backends)
if backends=='local_unitary_simulator':
    test_main = result.get_data('main')['unitary']
    test_err  = result.get_data('err')['unitary']
    qb_main = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
    qb_err  = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
    qb_main = np.matmul(test_main,qb_main)
    qb_err  = np.matmul(test_err ,qb_err )
    print(qb_main,'\n\n',qb_err)
    
elif backends=='local_qasm_simulator':
    test_main = result.get_data('main')['counts']
    test_err  = result.get_data('err')['counts']
    print(test_main)
    print(test_err)
    plot_histogram(test_main)
    plot_histogram(test_err)

test = True
if test==True:
    test_m = rdm.rdm(test_main)
    test_e = rdm.rdm(test_err)
    print(test_m,test_e)
    test_rdm, nocc, nvec = rdm.construct_rdm(test_m,test_e)
    print(test_rdm)
    print(nocc)


#print(test_main)
#plot_state(test_rho)
#plot_histogram(test_main)
#plot_histogram(test_err)
#print(result)
#print(result.get_counts('threeQ'))


