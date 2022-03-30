from hqca.acse import *
import pickle
import sys
import numpy as np
from qiskit.providers.aer import noise

from qiskit.providers.aer.noise.device.parameters import *
from qiskit.providers.aer.noise.device.models import *
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise.errors.readout_error import ReadoutError

class pseudo_properties:
    '''
    Creates a properties object from a IBMQ backend properties object.
    '''
    def __init__(self,properties):
        self.gates = properties.gates
        self.qubits = properties.qubits
        #print(dir(properties))
        #print(properties.qubits)

    def homogeneous(self,scaling=1.0):
        # gates
        gates = ['u1','u2','u3','cx','id']
        gate_length = {k:[0,0] for k in gates}
        gate_error = {k:[0,0] for k in gates}
        for g in self.gates:
            for p in g.parameters:
                t = g.gate
                if p.name=='gate_length':
                    gate_length[t][0]+= p.value*scaling
                    gate_length[t][1]+= 1
                if p.name=='gate_error':
                    gate_error[t][0]+= p.value*scaling
                    gate_error[t][1]+= 1
        # qubits
        # reassign 
        for g in self.gates:
            for p in g.parameters:
                t = g.gate
                if p.name=='gate_length':
                    p.value = gate_length[t][0]/gate_length[t][1]
                if p.name=='gate_error':
                    p.value = gate_error[t][0]/gate_error[t][1]
        for k,v in gate_error.items():
            gate_error[k]=v[0]/v[1]
        for k,v in gate_length.items():
            gate_error[k]=v[0]/v[1]
        qubits = [0,1,2,3,4]
        qubit_prob = {
                'T1':[0]*5,
                'T2':[0]*5,
                'frequency':[0]*5,
                'readout_error':[0]*5,
                'prob_meas0_prep1':[0]*5,
                'prob_meas1_prep0':[0]*5,
                }
        for n,q in enumerate(self.qubits):
            for p in q:
                #if p.name=='frequency':
                #    print(p)
                try:
                    qubit_prob[p.name][n]= p.value*scaling
                except KeyError:
                    pass
        for n,q in enumerate(self.qubits):
            for p in q:
                try:
                    p.value = np.average(qubit_prob[p.name])
                except KeyError:
                    pass

def noisy_model_from_ibmq(scaling=1.0):
    nm = noise.NoiseModel()
    name = 'ibmq_backend'
    try:
        with open(name,'rb') as fp:
            data = pickle.load(fp)
    except FileNotFoundError:
        print('Wrong one :(')
    prop = data['properties']
    n = pseudo_properties(prop)
    n.homogeneous(scaling=scaling)
    err = basic_device_gate_errors(properties=n)
    ro = basic_device_readout_errors(n)
    try:
        nm.add_all_qubit_readout_error(ro[0][1],'measure')
        nm.add_all_qubit_quantum_error(err[0][2],'id')
        nm.add_all_qubit_quantum_error(err[1][2],'u2')
        nm.add_all_qubit_quantum_error(err[2][2],'u3')
        nm.add_all_qubit_quantum_error(err[-1][2],'cx')
    except IndexError:
        pass
    return nm


def qiskit_average_noise_model(scaling=1.0):
    qs = QuantumStorage()
    name = 'ibmq_backend'
    try:
        with open(name,'rb') as fp:
            data = pickle.load(fp)
    except FileNotFoundError:
        print('Wrong one :(')
    prop = data['properties']

    be_coupling = data['config'].coupling_map
    gate_lengths = gate_length_values(prop)

    readout_prob =  np.zeros((2,2))
    basic_readout_errors = basic_device_readout_errors(prop)
    for p,e in basic_readout_errors:
        readout_prob+=np.asmatrix(e._probabilities)
    ro_prob = (readout_prob/(len(basic_readout_errors))).tolist()
    sq_readout = ReadoutError(ro_prob)
    thermal_errors = basic_device_gate_errors(
            prop,
            gate_error=True,
            thermal_relaxation=True,
            gate_lengths=gate_lengths,
            )
    tq_prob = np.zeros(9)
    id_prob = np.zeros(12)
    u2_prob = np.zeros(12)
    u3_prob = np.zeros(12)
    n = 0
    for a,b,c in thermal_errors:
        if a=='cx':
            if n==0:
                tq_circ = c._noise_circuits
            n+=1
            tq_prob+= np.asarray(c._noise_probabilities)
        elif a=='id':
            if b[0] in [0,'0']:
                id_circ = c._noise_circuits
            id_prob+= np.asarray(c._noise_probabilities)
        elif a=='u3':
            if b[0] in [0,'0']:
                u3_circ = c._noise_circuits
            u3_prob+= np.asarray(c._noise_probabilities)
        elif a=='u2':
            if b[0] in [0,'0']:
                u2_circ = c._noise_circuits
            u2_prob+= np.asarray(c._noise_probabilities)
    tq_prob = tq_prob/n
    u2_prob = u2_prob/5
    u3_prob = u3_prob/5
    id_prob = id_prob/5
    tq_noisy = []
    sq_id_noisy = []
    sq_u2_noisy = []
    sq_u3_noisy = []
    ro_prob = np.asmatrix(ro_prob)
    ro_prob[0,0]= 1+(ro_prob[0,0]-1)*scaling
    ro_prob[1,1]= 1+(ro_prob[1,1]-1)*scaling
    ro_prob[1,0]= (ro_prob[1,0])*scaling
    ro_prob[0,1]= (ro_prob[0,1])*scaling
    ro_prob = ro_prob.tolist()
    for p in [tq_prob,u2_prob,u3_prob,id_prob]:
        p[0] = p[0]-1
        p[:] = p*scaling
        p[0] = p[0]+1
    for op,p in zip(tq_circ,tq_prob):
        tq_noisy.append([op,p])
    for op,p in zip(id_circ,id_prob):
        sq_id_noisy.append([op,p])
    for op,p in zip(u2_circ,u2_prob):
        sq_u2_noisy.append([op,p])
    for op,p in zip(u3_circ,u3_prob):
        sq_u3_noisy.append([op,p])

    sq_ro_err = ReadoutError(ro_prob)
    tq_cx_err = QuantumError(tq_noisy)
    sq_id_err = QuantumError(sq_id_noisy)
    sq_u2_err = QuantumError(sq_u2_noisy)
    sq_u3_err = QuantumError(sq_u3_noisy)

    nm.add_all_qubit_readout_error(sq_ro_err,'measure')
    nm.add_all_qubit_quantum_error(sq_id_err,'id')
    nm.add_all_qubit_quantum_error(sq_u2_err,'u2')
    nm.add_all_qubit_quantum_error(sq_u3_err,'u3')
    nm.add_all_qubit_quantum_error(tq_cx_err,'cx')
    return nm

