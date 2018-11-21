'''
/tools/QuantumFramework.py

File for managing IBM process- note it does not actually load the IBM or qiskit
modules, but calls QuantumTomography, which does include those modules. Not
quite sure why though. 

'''

import time
import timeit
from hqca.tools.QuantumTomography import GenerateCompactTomography
from hqca.tools.QuantumTomography import GenerateDirectTomography
from hqca.tools.QuantumTomography import ProcessToRDM
from hqca.tools.IBM_check import check,get_backend_object
from hqca.tools.Functions import get_reading_material


#import qiskit.backends.local.qasm_simulator_cpp as qs

SIM_EXEC = ('/usr/local/lib/python3.5/dist-packages'
            '/qiskit/backends/qasm_simulator_cpp')

def evaluate(
        num_shots,
        split_runs=True,
        verbose=False,
        **kwargs
        ):
    '''
    Total list of kwargs:
    
    num_shots
    split_runs
    verbose
     
    GenerateTomography:
        connect
        **verbose
        **backend
        **algorithm
        **tomography
        _num_shots
        GenerateCircuit:
            **parameters
            algorithm
            **order
            _name
    ProcessToRDM:
        combine
    '''
    if split_runs:
        Nr = num_shots//1024
        num_shots = 1024
    else:
        Nr = 1
    kwargs['verbose']=verbose
    kwargs['_num_shots']=num_shots
    kwargs['_num_runs'] = Nr
    Data = ProcessToRDM(
            combine=split_runs
            )
    Data.add_data(
            GenerateTomography(
                **kwargs
                )
            )
    #for inst in range(0,Ns):
    #    qo = GenerateTomography(
    #            **kwargs
    #            )
    #    #print(qo)
    #    Data.add_instance(qo)
    Data.build_rdm(**kwargs)
    return Data

def add_to_config_log(backend,connect):
    '''
    Function to add the current ibm credentials and configuration to the
    ./results/logs/ directory, so it is stored for potential publications.
    '''
    # check if config file is already there
    from datetime import date
    import pickle
    today = date.timetuple(date.today())
    today = '{:04}{:02}{:02}'.format(today[0],today[1],today[2])
    loc = '/home/scott/Documents/research/3_vqa/hqca/results/logs/'
    filename = loc+today+'_'+backend
    from qiskit import get_backend,available_backends
    if connect:
        from qiskit import register
        try:
            import Qconfig
        except ImportError:
            from ibmqx import Qconfig
        try:
            register(Qconfig.APItoken)
        except Exception as e:
            #print(e)
            pass
    else:
        pass
    try:
        with open(filename,'rb') as fp:
            pass
        print('----------')
        print('Backend configuration file already written.')
        print('Stored in following location:')
        print('{}'.format(filename))
        print('----------')
    except FileNotFoundError:
        avail = available_backends()
        if backend in avail:
            pass
        elif backend=='ibmqx4':
            backend='ibmq_5_tenerife'
        with open(filename,'wb') as fp:
            data = pickle.dump(
                    get_backend(backend).calibration,
                    fp,0
                    )
        print('----------')
        print('Backend configuration file written.')
        print('Stored in following location:')
        print('{}'.format(filename))
        print('----------')

def wait_for_machine(backend,Nw=10):
    '''
    Function to wait for machine given a certain backend. 
    '''
    if backend=='ibmqx4':
        backend='ibmq_5_tenerife'
    qo = get_backend_object(backend)
    use,pending = check(qo)
    time_waited = 0
    while (use and pending>Nw):
        print('----------------------------------------')
        print('--- There are {:03} runs in queue. -----'.format(pending))
        print('--- Waited {:05} minutes so far. ----'.format((time_waited//60)))
        print('----------------------------------------')
        #get_reading_material()
        time.sleep(300) # wait for 5 minutes, check again
        time_waited+= 300
        try:
            use,pending = check(qo)
        except Exception as e:
            print('Might have run into an error.')
            print(e)
            print('Waiting a bit...3 minutes.')
            time.sleep(180)
            try:
                register(Qconfig.APItoken)
                use,pending = check(qo)
            except Exception:
                raise NotAvailableError
    if use:
        pass
    else:
        raise NotAvailableError

'''
backends = available_backends({'local': True})

test=get_backend('local_qasm_simulator')
shots = 1024
q = qiskit.QuantumRegister(2)
c = qiskit.ClassicalRegister(2)
qc = qiskit.QuantumCircuit(q,c)
qc.h(q[0])
qc.cx(q[1],q[0])
qc.measure(q,c)

q1 = qiskit.QuantumRegister(2)
c1= qiskit.ClassicalRegister(2)
qc1 = qiskit.QuantumCircuit(q1,c1)
qo = qiskit.compile(qc,test)
qc1.h(q1[0])
qc1.cx(q1[1],q1[0])
qc1.measure(q1,c1)


qo = qiskit.compile([qc,qc1],test)
print(qc.name)
qr =  test.run(qo)
print(qr)
print(qr.running)
time.sleep(1)
print(qr.running)

print(qr.result().get_counts('circuit1'))
print(test.configuration)
print(test.calibration)
print(test.parameters)

#for i,j in qr.items():
#    print(i,j)

'''








