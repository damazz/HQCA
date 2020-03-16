from hqca.acse import *
import sys
import numpy as np
from qiskit.providers.aer import noise

# procedure for generating noise model


from qiskit.providers.aer.noise.device.parameters import *
from qiskit.providers.aer.noise.device.models import *
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise.errors.readout_error import ReadoutError

def generateNoiseModel(scaling=1.0):
    qs = QuantumStorage()
    name = '/home/scott/Documents/research/'
    name+= '5_acse/runs/two_qubit/noisy/121619_ibmq_ourense'
    qs.set_noise_model(
            saved=name,
            )
    nm = noise.NoiseModel()
    prop = qs._be_properties
    gate_lengths = gate_length_values(prop)
    readout_prob =  np.zeros((2,2))
    basic_readout_errors = basic_device_readout_errors(prop)
    for p,e in basic_readout_errors:
        readout_prob+=np.asmatrix(e._probabilities)
    ro_prob = (readout_prob/(len(basic_readout_errors))).tolist()
    sq_readout = ReadoutError(ro_prob)
    thermal_errors = basic_device_gate_errors(
            qs._be_properties,
            gate_error=True,
            thermal_relaxation=True,
            gate_lengths=gate_lengths,
            )
    tq_prob = np.zeros(9)
    id_prob = np.zeros(3)
    u2_prob = np.zeros(3)
    u3_prob = np.zeros(3)
    
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

#generateNoiseModel(1.0)
#generateNoiseModel(0.5)
# now, calculation
