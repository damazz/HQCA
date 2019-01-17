'''
/tools/QuantumFramework.py

File for managing IBM process- note it does not actually load the IBM or qiskit
modules, but calls QuantumTomography, which does include those modules. Not
quite sure why though. 

'''

import time
import timeit
from hqca.tools._Tomography import Process
from hqca.tools._Tomography import local_qubit_tomo_pairs as lqtp
from hqca.tools._Tomography import nonlocal_qubit_tomo_pairs_full as nqtpf
from hqca.tools._Tomography import nonlocal_qubit_tomo_pairs_part as nqtpp
from hqca.tools.QuantumAlgorithms import GenerateDirectCircuit
from hqca.tools.QuantumAlgorithms import GenerateCompactCircuit
from hqca.tools.QuantumAlgorithms import algorithm_tomography
from hqca.tools.IBM_check import check,get_backend_object
import sys, traceback
from qiskit import Aer,IBMQ
from qiskit import compile as compileqp
from math import pi

SIM_EXEC = ('/usr/local/lib/python3.5/dist-packages'
            '/qiskit/backends/qasm_simulator_cpp')

def build_circuits(
        qa_fermion,
        pr_q=True,
        **kw
        ):
    def _init_compact(
            algorithm,
            **kw
            ):
        Nq = algorithm_tomography[algorithm]['Nq']
        qb_orbs = algorithm_tomography[algorithm]['qb_to_orb']
        No = len(qb_orbs)
        return Nq,qb_orbs,No

    def _init_direct(
            store,
            Nqb=1,
            qb2so='default',
            **kw,
            ):
        alpha = store.alpha_mo['qc']
        beta  = store.beta_mo['qc']
        so2qb,qb2so = _get_map(
                qb2so,alpha,beta
                )
        return alpha,beta,so2qb,qb2so
    def _get_map(
            qb2so,alpha,beta
            ):
        if qb2so=='default':
            so2qb = {}
            qb2so = {}
            for qb,so in enumerate(alpha):
                so2qb[so]=qb
                qb2so[qb]=so
            for qb,so in enumerate(beta):
                so2qb[so]=qb+len(alpha)
                qb2so[qb+len(alpha)]=so
        else:
            try:
                so2qb = {v:k for k,v in qb2so.items()}
            except Exception:
                traceback.print_exc()
        return so2qb,qb2so
    '''
    Begin actual circuit generation. 
    '''
    kw['pr_q']=pr_q
    if qa_fermion=='compact':
        Nq,qb_orbs,No = _init_compact(**kw)
        kw['Nq']=Nq
        kw['qb_orbs']=qb_orbs
        kw['No']=No
        q2s = {}
        circ, circ_list = _compact_tomography(**kw)
    elif qa_fermion=='direct':
        alp,bet, s2q,q2s = _init_direct(**kw)
        kw['alpha']=alp
        kw['beta']=bet
        kw['so2qb']=s2q
        kw['qb2so']=q2s
        circ, circ_list = _direct_tomography(**kw)
    return circ,circ_list,q2s


def _direct_tomography(
    alpha,
    beta,
    tomo_rdm='1rdm',
    tomo_basis='hada',
    tomo_extra=False,
    pr_q=0,
    **kw
        ):
    def _get_pairs(alp,bet):
        if tomo_basis=='hada':
            qtp = nqtpp
        elif tomo_basis=='sudo':
            qtp = lqtp
        elif tomo_basis=='pauli':
            qtp = nqtpf
        rdm_alp = qtp[len(alp)]
        temp = qtp[len(bet)]
        rdm_bet = []
        a = len(alp)
        for circ in temp:
            temp_arr = []
            for pair in circ:
                temp_arr.append('{}{}'.format(
                        str(int(pair[0])+a),
                        str(int(pair[1])+a)
                        )
                    )
            rdm_bet.append(temp_arr)
        circ_pairs = zip(rdm_alp,rdm_bet)
        return circ_pairs

    def apply_1rdm_basis_tomo(Q,i,k):
        '''
        generic 1rdm circuit for ses method
        '''
        # apply cz phase
        for l in range(i+1,k):
            Q.qc.cz(Q.q[i],Q.q[l])
        # apply cnot1
        Q.qc.cx(Q.q[k],Q.q[i])
        Q.qc.x(Q.q[k])
        # ch gate
        Q.qc.ry(pi/4,Q.q[k])
        Q.qc.cx(Q.q[i],Q.q[k])
        Q.qc.ry(-pi/4,Q.q[k])
        # apply cnot2
        Q.qc.x(Q.q[k])
        Q.qc.cx(Q.q[k],Q.q[i])
        return Q
    '''
    Begin direct tomography
    '''
    circuit,circuit_list = [],[]
    circ_pair = _get_pairs(alpha,beta)
    if tomo_rdm=='1rdm':
        if tomo_basis in ['no','NO']:
            print('Do you think we are in the natural orbitals? Wrong method!')
            sys.exit()
        elif tomo_basis=='hada':
            Q = GenerateDirectCircuit(**kw,_name='ii')
            Q.qc.measure(Q.q,Q.c)
            circuit.append(Q.qc)
            circuit_list.append(['ii'])
            i=0
            for ca,cb in circ_pair:
                Q = GenerateDirectCircuit(**kw,_name='ij{:02}'.format(i))
                temp = ['ij{:02}'.format(i)]
                for pair in ca:
                    Q = apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]))
                    temp.append(pair)
                for pair in cb:
                    Q = apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]))
                    temp.append(pair)
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append(temp)
                circuit.append(Q.qc)
                '''
                Q = GenerateDirectCircuit(**kw,_name='iZj{}'.format(i))
                temp = ['iZj{}'.format(i)]
                for pair in ca:
                    Q = apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]),
                            Z=True)
                    temp.append(pair)
                for pair in cb:
                    Q = apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]),
                            Z=True)
                    temp.append(pair)
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append(temp)
                circuit.append(Q.qc)
                '''
                i+=1 
        elif tomo_basis=='pauli':
            # projection into the pauli basis now...
            pass
    if tomo_extra:
        # looking at tomo of RDM with 
        pass
    return circuit,circuit_list

def _compact_tomography(
        qb_orbs,
        tomo_rdm,
        tomo_basis,
        **kw
        ):
    def get_qb_parity(
            qb_orbs,
            ):
        ind = 0
        qb_sign = {}
        for item in reversed(qb_orbs):
            qb_sign[item]=(-1)**ind
            ind+=1
        return qb_sign
    '''
    '''
    circuit,circuit_list = [],[]
    if tomo_rdm=='1rdm' and tomo_basis=='no':
        qb_sign = get_qb_parity(qb_orbs)
        Q = GenerateCompactCircuit(**kw,_name='ii')
        Q.qc.measure(Q.q,Q.c)
        circuit.append(Q.qc)
        circuit_list.append(['ii'])
    elif tomo_rdm=='1rdm' and tomo_basis=='bch':
        Q = GenerateCompactCircuit(**kw,_name='ii')
        Q.qc.measure(Q.q,Q.c)
        circuit_list.append(['ii'])
        circuit.append(self.Q[i].qc)
        Q = GenerateCircuit(**kw,_name='ij')
        for j in qb_orbs:
            if qb_sign[j]==-1:
                Q.qc.z(Q.q[j])
                Q.qc.h(Q.q[j])
            elif self.qb_sign[j]==1:
                Q.qc.h(Q.q[j])
                Q.qc.z(Q.q[j])
            else:
                sys.exit('Error in performing 1RDM tomography.')
        Q.qc.measure(Q.q,Q.c)
        circuit_list.append(['ij'])
        circuit.append(Q.qc)
    elif tomo_rdm=='2rdm':
        print('Why are you doing 2-RDM tomography on a compact system?')
        print('Unnecessary!')
        sys.exit()
    return circuit, circuit_list




def run_circuits(
        circuits,
        circuit_list,
        qc_backend,
        qc_provider,
        qc_num_shots,
        pr_q=0,
        **kw
        ):
    def _tomo_get_backend(
            provider,
            backend
            ):
        if provider=='Aer':
            prov=Aer
        elif provider=='IBMQ':
            prov=IBMQ
        try:
            return prov.get_backend(backend)
        except Exception:
            traceback.print_exc()
    beo = _tomo_get_backend(
            qc_provider,
            qc_backend)
    #for i in circuits:
    #    print(i.qasm())
    #    print('')
    qo = compileqp(
            circuits,
            shots=qc_num_shots,
            backend=beo
            )
    counts = []
    if qc_backend=='local_unitary_simulator':
        qr = beo.run(qo,timeout=6000)
        for circ in circuit_list:
            U = qr.result().get_data(circ)['unitary']
            counts[circuit] = fx.UnitaryToCounts(U)
    else:
        try:
            job = beo.run(qo)
            for circuit in circuit_list:
                name = circuit[0]
                counts.append(job.result().get_counts(name))
        except Exception as e:
            print('Error: ')
            print(e)
            traceback.print_exc()
    #if verbose:
    #    print('Circuit counts:')
    #    for i in counts:
    #        for k,v in i.items():
    #            print('  {}:{}'.format(k,v))
    return zip(circuit_list,counts)

def construct(
        data,
        **kw):
    qo = Process(data,**kw)
    qo.build_rdm()
    return qo.rdm








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








