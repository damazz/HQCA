from qiskit import QuantumProgram, QuantumCircuit
#from qiskit.tools.visualization import plot_histogram, plot_state
import numpy as np
from . import rdm
#from qiskit.tools.qi.qi import state_fidelity,concurrence,purity,outer
import traceback
import pprint
np.set_printoptions(precision=4,linewidth=200)
tf_ibm_qx2 = {'01':True,'02':True, '12':True, '10':False,'20':False, '21':False}

tf_ibm_qx4 = {'01':False,'02':False, '12':False, '10':True,'20':True, '21':True}




def single_run_alpha(theta1,theta2,theta3,order,accu,API_token,API_config,backend_input):
    qp = QuantumProgram()
    try:
        qp.set_api(API_token,API_config)
    except Exception: 
        pass
    qr_main = qp.create_quantum_register("main_qr",3)
    qr_err  = qp.create_quantum_register("err_qr", 3)
    cr_main = qp.create_classical_register("main_cr",3)
    cr_err  = qp.create_classical_register("err_cr",3)
    main    = qp.create_circuit('main',[qr_main],[cr_main])
    err     = qp.create_circuit('err',[qr_err],[cr_err])
    
    main.ry(2*theta1,qr_main[order[0]])
    main.cx(qr_main[order[0]],qr_main[order[1]]) 
    main.ry(2*theta2,qr_main[order[2]])
    main.cx(qr_main[order[2]],qr_main[order[3]]) 
    main.ry(2*theta3, qr_main[order[4]])
    main.cx(qr_main[order[4]],qr_main[order[5]]) 
    main.measure(qr_main[0],cr_main[0])
    main.measure(qr_main[1],cr_main[1])
    main.measure(qr_main[2],cr_main[2])
  
    err.ry(2*theta1,qr_err[order[0]])
    err.cx(qr_err[order[0]],qr_err[order[1]]) 
    err.ry(2*theta2,qr_err[order[2]])
    err.cx(qr_err[order[2]],qr_err[order[3]]) 
    err.ry(2*theta3,qr_err[order[4]])
    err.cx(qr_err[order[4]],qr_err[order[5]]) 
    err.ry(np.pi/2,qr_err[0])
    err.ry(-np.pi/2,qr_err[1])
    err.ry(np.pi/2,qr_err[2])
    err.measure(qr_err[0],cr_err[0])
    err.measure(qr_err[1],cr_err[1])
    err.measure(qr_err[2],cr_err[2])
    circuits = ['main','err']
     
    results = qp.execute(circuits,shots=accu,backend=backend_input)
    return results, circuits

def single_run_beta(theta1,theta2,theta3,order,accu,API_token,API_config,backend_input):
    # includes the reveresed CNOT gate for the reveresed order
    qp = QuantumProgram()
    try:
        qp.set_api(API_token,API_config)
    except Exception: 
        pass
    qr_main = qp.create_quantum_register("main_qr",3)
    qr_err  = qp.create_quantum_register("err_qr", 3)
    cr_main = qp.create_classical_register("main_cr",3)
    cr_err  = qp.create_classical_register("err_cr",3)
    main    = qp.create_circuit('main',[qr_main],[cr_main])
    err     = qp.create_circuit('err',[qr_err],[cr_err])
    
    main.ry(2*theta1,qr_main[order[0]])
    main.cx(qr_main[order[0]],qr_main[order[1]]) 
    main.ry(2*theta2,qr_main[order[2]])
    main.cx(qr_main[order[2]],qr_main[order[3]]) 
    main.ry(2*theta3, qr_main[order[4]])
    if order[4]>order[5]:
        main.h(qr_main[order[4]])
        main.h(qr_main[order[5]])
        main.cx(qr_main[order[5]],qr_main[order[4]]) 
        main.h(qr_main[order[4]])
        main.h(qr_main[order[5]])
    else:
        main.cx(qr_main[order[4]],qr_main[order[5]]) 
    main.measure(qr_main[0],cr_main[0])
    main.measure(qr_main[1],cr_main[1])
    main.measure(qr_main[2],cr_main[2])
  
    err.ry(2*theta1,qr_err[order[0]])
    err.cx(qr_err[order[0]],qr_err[order[1]]) 
    err.ry(2*theta2,qr_err[order[2]])
    err.cx(qr_err[order[2]],qr_err[order[3]]) 
    err.ry(2*theta3,qr_err[order[4]])
    if order[4]>order[5]:
        err.h(qr_err[order[4]])
        err.h(qr_err[order[5]])
        err.cx(qr_err[order[5]],qr_err[order[4]]) 
        err.h(qr_err[order[4]])
        err.h(qr_err[order[5]])
    else:
        err.cx(qr_err[order[4]],qr_err[order[5]]) 
    err.ry(np.pi/2,qr_err[0])
    err.ry(-np.pi/2,qr_err[1])
    err.ry(np.pi/2,qr_err[2])
    err.measure(qr_err[0],cr_err[0])
    err.measure(qr_err[1],cr_err[1])
    err.measure(qr_err[2],cr_err[2])
    circuits = ['main','err']
     
    results = qp.execute(circuits,shots=accu,backend=backend_input)
    return results, circuits



def single_run_gamma(theta1,theta2,theta3,order,accu,API_token,API_config,backend_input):
    # have a two qubit gate to check error along the GPC vertex  
    # also, configured for ibmqx4 and ibmqx2
    qp = QuantumProgram()
    try:
        qp.set_api(API_token,API_config)
    except Exception: 
        print('API error in single run gamma.')
        pass
    qr_main = qp.create_quantum_register("main_qr",3)
    qr_err  = qp.create_quantum_register("err_qr", 3)
    cr_main = qp.create_classical_register("main_cr",3)
    cr_err  = qp.create_classical_register("err_cr",3)
    main    = qp.create_circuit('main',[qr_main],[cr_main])
    err     = qp.create_circuit('err',[qr_err],[cr_err])

    #
    # check if the orering of each qubit CNOT is allowed
    # i.e., set conditioning
    # 

    c1 = str(order[0])+str(order[1])
    c2 = str(order[2])+str(order[3])
    c3 = str(order[4])+str(order[5])
    if backend_input=='ibmqx2':
        c1 = tf_ibm_qx2[c1]  
        c2 = tf_ibm_qx2[c2]  
        c3 = tf_ibm_qx2[c3]  
    elif backend_input=='ibmqx4':
        c1 = tf_ibm_qx4[c1]  
        c2 = tf_ibm_qx4[c2]  
        c3 = tf_ibm_qx4[c3]  
    else:
        c1 = True
        c2 = True
        c3 = True
    #
    # start the main circuit
    #

    main.ry(2*theta1,qr_main[order[0]])
    if c1:
        main.cx(qr_main[order[0]],qr_main[order[1]]) 
    else:
        main.h(qr_main[order[0]])
        main.h(qr_main[order[1]])
        main.cx(qr_main[order[1]],qr_main[order[0]]) 
        main.h(qr_main[order[0]])
        main.h(qr_main[order[1]])     

    main.ry(2*theta2,qr_main[order[2]])
    if c2:
        main.cx(qr_main[order[2]],qr_main[order[3]]) 
    else:
        main.h(qr_main[order[2]])
        main.h(qr_main[order[3]])
        main.cx(qr_main[order[3]],qr_main[order[2]]) 
        main.h(qr_main[order[2]])
        main.h(qr_main[order[3]])

    main.ry(2*theta3, qr_main[order[4]])
    if c3:
        main.cx(qr_main[order[4]],qr_main[order[5]]) 
    else:
        main.h(qr_main[order[4]])
        main.h(qr_main[order[5]])
        main.cx(qr_main[order[5]],qr_main[order[4]]) 
        main.h(qr_main[order[4]])
        main.h(qr_main[order[5]])

    main.measure(qr_main[0],cr_main[0])
    main.measure(qr_main[1],cr_main[1])
    main.measure(qr_main[2],cr_main[2])
 
    #
    # start the err circuit
    #
  
    err.ry(2*theta1,qr_err[order[0]])
    if c1:
        err.cx(qr_err[order[0]],qr_err[order[1]]) 
    else:
        err.h(qr_err[order[0]])
        err.h(qr_err[order[1]])
        err.cx(qr_err[order[1]],qr_err[order[0]]) 
        err.h(qr_err[order[0]])
        err.h(qr_err[order[1]])
     
    err.ry(2*theta2,qr_err[order[2]])
    if c2:
        err.cx(qr_err[order[2]],qr_err[order[3]]) 
    else:
        err.h(qr_err[order[2]])
        err.h(qr_err[order[3]])
        err.cx(qr_err[order[3]],qr_err[order[2]]) 
        err.h(qr_err[order[2]])
        err.h(qr_err[order[3]])

    err.ry(2*theta3, qr_err[order[4]])
    if c3:
        err.cx(qr_err[order[4]],qr_err[order[5]]) 
    else:
        err.h(qr_err[order[4]])
        err.h(qr_err[order[5]])
        err.cx(qr_err[order[5]],qr_err[order[4]]) 
        err.h(qr_err[order[4]])
        err.h(qr_err[order[5]])

    # tomography qubit transformations

    err.ry(np.pi/2,qr_err[0])
    err.ry(-np.pi/2,qr_err[1])
    err.ry(np.pi/2,qr_err[2])

    err.measure(qr_err[0],cr_err[0])
    err.measure(qr_err[1],cr_err[1])
    err.measure(qr_err[2],cr_err[2])

    circuits = ['main','err']
     
    results = qp.execute(circuits,shots=accu,backend=backend_input,wait=5,timeout=600,silent=False)
    return results, circuits


def single_run_delta(theta1,theta2,order,accu,API_token,API_config,backend_input):
    # have a two qubit gate to check error along the GPC vertex  
    # also, configured for ibmqx4 and ibmqx2
    qp = QuantumProgram()
    try:
        qp.set_api(API_token,API_config)
    except Exception: 
        pass
    qr_main = qp.create_quantum_register("main_qr",3)
    qr_err  = qp.create_quantum_register("err_qr", 3)
    cr_main = qp.create_classical_register("main_cr",3)
    cr_err  = qp.create_classical_register("err_cr",3)
    main    = qp.create_circuit('main',[qr_main],[cr_main])
    err     = qp.create_circuit('err',[qr_err],[cr_err])

    #
    # check if the orering of each qubit CNOT is allowed
    # i.e., set conditioning
    # 

    c1 = str(order[0])+str(order[1])
    c2 = str(order[2])+str(order[3])
    if backend_input=='ibmqx2':
        c1 = tf_ibm_qx2[c1]  
        c2 = tf_ibm_qx2[c2]  
    elif backend_input=='ibmqx4':
        c1 = tf_ibm_qx4[c1]  
        c2 = tf_ibm_qx4[c2]  
    else:
        c1 = True
        c2 = True
    #
    # start the main circuit
    #

    main.ry(2*theta1,qr_main[order[0]])
    if c1:
        main.cx(qr_main[order[0]],qr_main[order[1]]) 
    else:
        main.h(qr_main[order[0]])
        main.h(qr_main[order[1]])
        main.cx(qr_main[order[1]],qr_main[order[0]]) 
        main.h(qr_main[order[0]])
        main.h(qr_main[order[1]])     

    main.ry(2*theta2,qr_main[order[2]])
    if c2:
        main.cx(qr_main[order[2]],qr_main[order[3]]) 
    else:
        main.h(qr_main[order[2]])
        main.h(qr_main[order[3]])
        main.cx(qr_main[order[3]],qr_main[order[2]]) 
        main.h(qr_main[order[2]])
        main.h(qr_main[order[3]])

    main.measure(qr_main[0],cr_main[0])
    main.measure(qr_main[1],cr_main[1])
    main.measure(qr_main[2],cr_main[2])
 
    #
    # start the err circuit
    #
  
    err.ry(2*theta1,qr_err[order[0]])
    if c1:
        err.cx(qr_err[order[0]],qr_err[order[1]]) 
    else:
        err.h(qr_err[order[0]])
        err.h(qr_err[order[1]])
        err.cx(qr_err[order[1]],qr_err[order[0]]) 
        err.h(qr_err[order[0]])
        err.h(qr_err[order[1]])
     
    err.ry(2*theta2,qr_err[order[2]])
    if c2:
        err.cx(qr_err[order[2]],qr_err[order[3]]) 
    else:
        err.h(qr_err[order[2]])
        err.h(qr_err[order[3]])
        err.cx(qr_err[order[3]],qr_err[order[2]]) 
        err.h(qr_err[order[2]])
        err.h(qr_err[order[3]])

    # tomography qubit transformations

    err.ry(np.pi/2,qr_err[0])
    err.ry(-np.pi/2,qr_err[1])
    err.ry(np.pi/2,qr_err[2])

    err.measure(qr_err[0],cr_err[0])
    err.measure(qr_err[1],cr_err[1])
    err.measure(qr_err[2],cr_err[2])

    circuits = ['main','err']
     
    results = qp.execute(circuits,shots=accu,backend=backend_input)
    return results, circuits

def con_epsilon(theta1,theta2,theta3,order,accu,API_token,API_config,backend_input,calc_od=True):
    # from gamma and delta format, so can support all 5 qubit or simulator backends
    # changes the output however, so that instead of an occupation number format, you have a 
    # diagonal element format, which allows one to exactly put together the 1-RDM
    # Also, there is a check for calculating the off-diagonal elements
    qp = QuantumProgram()
    try:
        qp.set_api(API_token,API_config)
    except Exception: 
        pass
    qr_main = qp.create_quantum_register("main_qr",3)
    cr_main = qp.create_classical_register("main_cr",3)
    main    = qp.create_circuit('main',[qr_main],[cr_main])

    if calc_od==True:
        qr_err  = qp.create_quantum_register("err_qr", 3)
        cr_err  = qp.create_classical_register("err_cr",3)
        err     = qp.create_circuit('err',[qr_err],[cr_err])

    #
    # check if the orering of each qubit CNOT is allowed
    # i.e., set conditioning
    # 

    c1 = str(order[0])+str(order[1])
    c2 = str(order[2])+str(order[3])
    c3 = str(order[4])+str(order[5])
    if backend_input=='ibmqx2':
        c1 = tf_ibm_qx2[c1]  
        c2 = tf_ibm_qx2[c2]  
        c3 = tf_ibm_qx2[c3]  
    elif backend_input=='ibmqx4':
        c1 = tf_ibm_qx4[c1]  
        c2 = tf_ibm_qx4[c2]  
        c3 = tf_ibm_qx4[c3]  
    else:
        c1 = True
        c2 = True
        c3 = True
    #
    # start the main circuit
    #

    main.ry(2*theta1,qr_main[order[0]])
    if c1:
        main.cx(qr_main[order[0]],qr_main[order[1]]) 
    else:
        main.h(qr_main[order[0]])
        main.h(qr_main[order[1]])
        main.cx(qr_main[order[1]],qr_main[order[0]]) 
        main.h(qr_main[order[0]])
        main.h(qr_main[order[1]])     

    main.ry(2*theta2,qr_main[order[2]])
    if c2:
        main.cx(qr_main[order[2]],qr_main[order[3]]) 
    else:
        main.h(qr_main[order[2]])
        main.h(qr_main[order[3]])
        main.cx(qr_main[order[3]],qr_main[order[2]]) 
        main.h(qr_main[order[2]])
        main.h(qr_main[order[3]])

    main.ry(2*theta3, qr_main[order[4]])
    if c3:
        main.cx(qr_main[order[4]],qr_main[order[5]]) 
    else:
        main.h(qr_main[order[4]])
        main.h(qr_main[order[5]])
        main.cx(qr_main[order[5]],qr_main[order[4]]) 
        main.h(qr_main[order[4]])
        main.h(qr_main[order[5]])

    main.measure(qr_main[0],cr_main[0])
    main.measure(qr_main[1],cr_main[1])
    main.measure(qr_main[2],cr_main[2])
 
    #
    # start the err circuit
    #
    if calc_od==True:
        err.ry(2*theta1,qr_err[order[0]])
        if c1:
            err.cx(qr_err[order[0]],qr_err[order[1]]) 
        else:
            err.h(qr_err[order[0]])
            err.h(qr_err[order[1]])
            err.cx(qr_err[order[1]],qr_err[order[0]]) 
            err.h(qr_err[order[0]])
            err.h(qr_err[order[1]])
         
        err.ry(2*theta2,qr_err[order[2]])
        if c2:
            err.cx(qr_err[order[2]],qr_err[order[3]]) 
        else:
            err.h(qr_err[order[2]])
            err.h(qr_err[order[3]])
            err.cx(qr_err[order[3]],qr_err[order[2]]) 
            err.h(qr_err[order[2]])
            err.h(qr_err[order[3]])

        err.ry(2*theta3, qr_err[order[4]])
        if c3:
            err.cx(qr_err[order[4]],qr_err[order[5]]) 
        else:
            err.h(qr_err[order[4]])
            err.h(qr_err[order[5]])
            err.cx(qr_err[order[5]],qr_err[order[4]]) 
            err.h(qr_err[order[4]])
            err.h(qr_err[order[5]])

        # tomography qubit transformations

        err.ry(np.pi/2,qr_err[0])
        err.ry(-np.pi/2,qr_err[1])
        err.ry(np.pi/2,qr_err[2])

        err.measure(qr_err[0],cr_err[0])
        err.measure(qr_err[1],cr_err[1])
        err.measure(qr_err[2],cr_err[2])

        circuits = ['main','err']
    else:
        circuits = ['main'] 
    results = qp.execute(circuits,shots=accu,backend=backend_input,wait=5,timeout=600,silent=False)
    return results, circuits


#################################

#################################

#################################


def multi_run_alpha(algorithm,n_qubits,the1,the2,the3,ordering,shots,config,url,backend):
    size = len(the1)*len(the2)*len(the3)*len(ordering)
    ind = 0 
    hold = np.zeros((size,6))
    error = np.zeros((size,3))
    for one in the1:
        for two in the2:
            for thr in the3:
                for ords in ordering:
                    try:
                        if algorithm=='single_run_beta':
                            results, circuits = single_run_beta(one,two,thr,ords,shots,config,url,backend)
                        elif algorithm=='single_run_alpha':
                            results, circuits = single_run_alpha(one,two,thr,ords,shots,config,url,backend)
                        elif algorithm=='single_run_gamma':
                            try:
                                results, circuits = single_run_gamma(one,two,thr,ords,shots,config,url,backend)
                            except:
                                traceback.print_exc()
                                print('Something wrong with the compiled circuit.')
                        elif algorithm=='single_run_delta':
                            results, circuits = single_run_delta(one,two,ords,shots,config,url,backend)
                        if backend=='local_unitary_simulator':
                            use_unitary=True
                            res_type   ='unitary'
                        else:
                            use_unitary=False
                            res_type   ='counts'
                        print(results.__str__())
                        print(results.get_error())
                        state_main = results.get_data(circuits[0])[res_type]
                        state_err  = results.get_data(circuits[1])[res_type]
                        main_r = rdm.rdm(state_main,unitary=use_unitary)
                        err_r  = rdm.rdm(state_err,unitary=use_unitary)
                        main_r = main_r[n_qubits-3:n_qubits]
                        err_r  = err_r[n_qubits-3:n_qubits]
                        rdm_r, occ_r, vec_r = rdm.construct_rdm(main_r,err_r)
                        print(rdm_r)
                        hold[ind,:] = occ_r[:]
                        error[ind,:] = err_r[:]-0.5
                        print('T1={:.1f}, T2={:.1f}, T3={:.3f}'.format(one*180/np.pi,two*180/np.pi,thr*180/np.pi))
                        print('Occupation numbers: ',occ_r,'\n','Error elements: ',err_r-0.5)
                        ind += 1
                        print('{:.3f}'.format(100*ind/size),'%','\n') 
                    except:
                        hold = hold[0:ind,:]
                        error  = error[0:ind,:]
                        print('Some error in multi run alpha.')
                        traceback.print_exc()
                        return hold, error, ind
 
    return hold, error, ind


def multi_run_beta(algorithm,n_qubits,the1,the2,the3,ordering,shots,config,url,backend,calc_od=True,combine=False):
    # designed for construct algorithms epsilon, and a different data formatting which does not involve 
    # constructing the RDM and eigenvalues, but instead outputs the values
    # is backwards compatible
    size = len(the1)*len(the2)*len(the3)*len(ordering)
    ind = 0 
    data = []
    n_run = range(0,shots//1024)
    if combine==False:
        n_run = [shots]
    elif combine==True:
        shots=1024
    p1 = 0
    for one in the1:
        p2 = 0
        for two in the2:
            p3 = 0
            for thr in the3:
                ors = 0 
                for ords in ordering:
                    nr = 0
                    for num in n_run:
                        if algorithm=='con_epsilon':
                            try:
                                results,circuits = con_epsilon(one,two,thr,ords,shots,config,url,backend,calc_od)
                                if backend=='local_unitary_simulator':
                                    use_unitary=True
                                    res_type   ='unitary'
                                else:
                                    use_unitary=False
                                    res_type   ='counts'
                                cr = 0
                                for item in circuits:
                                    data.append([])
                                    data[ind].append('C{}N{}Q{}P{}{}{}'.format(cr,nr,ors,p1,p2,p3))
                                    data[ind].append(results.get_data(item)[res_type])
                                    cr+=1
                                    ind +=1 
                            except:
                                traceback.print_exc()
                                return data,ind 
                        else:
                            sys.exit('Wrong algorithm being used')
                        nr+= 1
                    ors+= 1
                p3 += 1
            p2+=1 
        p1+= 1
    return data,ind
