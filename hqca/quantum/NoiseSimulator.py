'''
quantum/NoiseSimulator.py

Contains information for loading, handling, and constructing noise simulators
with qiskit. Ideally, would be more general and would have different components,
but this currently is not the case. 


'''
import sys
from qiskit import Aer,IBMQ
import pickle
import traceback
from qiskit.tools.monitor import job_monitor
from qiskit import execute
import qiskit
from qiskit.compiler import assemble
from qiskit.providers.aer import noise
from hqca.ibm import Qconfig
import hqca
from qiskit.ignis.mitigation.measurement import(
        complete_meas_cal,
        tensored_meas_cal,
        TensoredMeasFitter,
        CompleteMeasFitter,
        MeasurementFilter)




def get_measurement_filter(
        QuantStore
        ):
    try:
        be = IBMQ.get_backend(QuantStore.backend)
    except Exception as e:
        traceback.print_exc()
    qubit_list= [i for i in range(QuantStore.Nq_tot)]
    if QuantStore.pr_e>0:
        print('Obtaining measurement filter.')
    cal_circuits,state_labels = complete_meas_cal(
            qubit_list,
            qiskit.QuantumRegister(QuantStore.Nq_tot),
            qiskit.ClassicalRegister(QuantStore.Nq_tot)
            )
    '''
    qo = assemble(
            cal_circuits,
            shots=QuantStore.Ns
            )
    '''
    job = execute(cal_circuits,
            backend=be,
            shots=QuantStore.Ns,
            initial_layout=QuantStore.be_initial)
    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(
            cal_results,
            state_labels)
    '''
    elif QuantStore.backend=='ibmq_16_melbourne':
        print(QuantStore.be_initial)
        if QuantStore.pr_e>0:
            print('Obtaining measurement filter.')
        cal_circuits,state_labels = tensored_meas_cal(
                mit_pattern=[qubit_list],
                qr=qiskit.QuantumRegister(14),
                cr=qiskit.ClassicalRegister(14)
                )
        qo = assemble(
                cal_circuits,
                shots=QuantStore.Ns
                )
        job = be.run(qo)
        cal_results = job.result()
        meas_fitter = TensoredMeasFitter(
                cal_results,
                state_labels)
        print(meas_fitter.cal_matrices)
        '''
    meas_filter = meas_fitter.filter
    QuantStore.meas_fitter = meas_fitter
    QuantStore.meas_filter = meas_filter
    if QuantStore.pr_e>0:
        print('Measurement filter complete!')



def get_noise_model(device,times=None,saved=False):
    if (not saved) or (saved is None):
        IBMQ.load_accounts()
        backend = IBMQ.get_backend(device)
        properties = backend.properties()
    else:
        try:
            with open(saved,'rb') as fp:
                data = pickle.load(fp)
        except FileNotFoundError:
            print('Wrong one :(')
        properties = data['properties']
    if times is not None:
        noise_model = noise.device.basic_device_noise_model(
            properties,times)
    else:
        noise_model = noise.device.basic_device_noise_model(
            properties)
    noise_model.coupling_map = data['config'].coupling_map
    return noise_model

def get_coupling_map(device,saved=True):
    if (not saved) or (saved is None):
        IBMQ.load_accounts()
        backend = IBMQ.get_backend(device)
        coupling = backend.configuration
    else:
        try:
            with open(saved,'rb') as fp:
                data = pickle.load(fp)
        except FileNotFoundError:
            print('Wrong one :(')
        coupling = data['config'].coupling_map
    return coupling
