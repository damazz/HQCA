'''
/tools/QuantumFramework.py

File for managing IBM process- note it does not actually load the IBM or qiskit
modules, but is involved with , which does include those modules. Not
quite sure why though.

'''

import time
import timeit
from itertools import zip_longest
from hqca.quantum.QuantumProcess import Process
from hqca.quantum.QuantumFunctions import local_qubit_tomo_pairs as lqtp
from hqca.quantum.QuantumFunctions import nonlocal_qubit_tomo_pairs_full as nqtpf
from hqca.quantum.QuantumFunctions import nonlocal_qubit_tomo_pairs_part as nqtpp
from hqca.quantum.QuantumFunctions import diag
from hqca.quantum.BuildCircuit import GenerateDirectCircuit
from hqca.quantum.BuildCircuit import GenerateCompactCircuit
from hqca.quantum.algorithms import _Tomo as tomo
import sys, traceback
from qiskit import Aer,IBMQ,execute
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from qiskit.tools.monitor import backend_overview,job_monitor
from hqca.quantum.NoiseSimulator import get_noise_model,get_coupling_map
from math import pi
from sympy import pprint


class Construct:
    def __init__(self,data,QuantStore):
        self.qo = Process(data,QuantStore)
        self.qo.build_rdm()
        self.rdm1 = self.qo.rdm

    def find_signs(self):
        self.qo.sign_from_2rdm()
        self.signs = self.qo.sign
        self.holding = self.qo.holding


def build_circuits(
        QuantStore,
        **kw
        ):
    '''
    Given certain instructions, will actually build the circuits. This is
    initial function that is called from other modules.
    '''
    if QuantStore.fermion_mapping=='compact':
        sys.exit('Please configure compact. Why.' )
        circ, circ_list = _compact_tomography(**kw)
    elif QuantStore.fermion_mapping=='jordan-wigner':
        circ, circ_list = _direct_tomography(QuantStore)
    else: 
        sys.exit(
            'Unsupported mapping of fermions: {}'.format(
                QuantStore.fermion_mapping)
                )
    return circ,circ_list

def _direct_tomography(
        QuantStore,
        ):
    '''
    Called by build_circuits(), will generate the tomography circuits of the 
    '''
    def _get_pairs(tomo_basis,alp,bet):
        if tomo_basis in ['hada','hada+imag']:
            qtp = nqtpp
        elif tomo_basis=='sudo':
            qtp = lqtp
        elif tomo_basis=='pauli':
            qtp = nqtpf
        else:
            qtp = diag
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
        circ_pairs = zip_longest(rdm_alp,rdm_bet)
        return circ_pairs
    '''
    Begin direct tomography
    '''
    circuit,circuit_list = [],[]
    circ_pair = _get_pairs(
            QuantStore.tomo_bas,
            QuantStore.alpha_qb,
            QuantStore.beta_qb)
    if QuantStore.tomo_rdm=='1rdm':
        if QuantStore.tomo_bas in ['no','NO']:
            Q = GenerateDirectCircuit(QuantStore,_name='ii')
            Q.qc.measure(Q.q,Q.c)
            print(Q.qc)
            circuit.append(Q.qc)
            circuit_list.append(['ii'])
        elif QuantStore.tomo_bas=='hada':
            Q = GenerateDirectCircuit(QuantStore,_name='ii')
            Q.qc.measure(Q.q,Q.c)
            circuit.append(Q.qc)
            circuit_list.append(['ii'])
            i=0
            for ca,cb in circ_pair:
                Q = GenerateDirectCircuit(QuantStore,_name='ijR{:02}'.format(i))
                temp = ['ijR{:02}'.format(i)]
                for pair in ca:
                    Q = apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]))
                    temp.append(pair)
                try:
                    for pair in cb:
                        Q = apply_1rdm_basis_tomo(
                                Q,int(pair[0]),int(pair[1]))
                        temp.append(pair)
                except TypeError as e:
                    print(e)
                except Exception as e:
                    traceback.print_exc()
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append(temp)
                circuit.append(Q.qc)
                i+=1 
        elif QuantStore.tomo_bas=='hada+imag':
            Q = GenerateDirectCircuit(QuantStore,_name='ii')
            Q.qc.measure(Q.q,Q.c)
            circuit.append(Q.qc)
            circuit_list.append(['ii'])
            i=0
            for ca,cb in circ_pair:
                Q = GenerateDirectCircuit(QuantStore,_name='ijR{:02}'.format(i))
                temp = ['ijR{:02}'.format(i)]
                for pair in ca:
                    apply_1rdm_basis_tomo(
                        Q,int(pair[0]),int(pair[1]))
                    temp.append(pair)
                try:
                    for pair in cb:
                        apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]))
                        temp.append(pair)
                except TypeError as e:
                    print(e)
                except Exception as e:
                    traceback.print_exc()
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append(temp)
                circuit.append(Q.qc)
                i+=1 
            circ_pair = _get_pairs(
                    QuantStore.tomo_bas,
                    QuantStore.alpha_qb,
                    QuantStore.beta_qb)
            for ca,cb in circ_pair:
                Q = GenerateDirectCircuit(QuantStore,_name='ijI{:02}'.format(i))
                temp = ['ijI{:02}'.format(i)]
                for pair in ca:
                    Q = apply_1rdm_basis_tomo(
                            Q,int(pair[0]),int(pair[1]),imag=True)
                    temp.append(pair)
                try:
                    for pair in cb:
                        Q = apply_1rdm_basis_tomo(
                                Q,int(pair[0]),int(pair[1]),imag=True)
                        temp.append(pair)
                except TypeError:
                    pass
                except Exception:
                    print('Error')
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append(temp)
                circuit.append(Q.qc)
                i+=1
        elif QuantStore.tomo_bas=='pauli':
            # projection into the pauli basis now...
            pass
    if QuantStore.tomo_ext=='sign_2e_pauli':
        n = 0 
        for a,b,c,d in QuantStore.qc_tomo_quad:
            if QuantStore.tomo_approx=='full':
                operators = [
                        'xxxx','xxyy','xyxy','xyyx',
                        'yxxy','yxyx','yyxx','yyyy']
            elif QuantStore.tomo_approx=='fo':
                operators = ['xxxx','yyyy']
            elif QuantStore.tomo_approx=='so':
                operators = ['xxxx','xxyy','yyxx','yyyy']
            else:
                print('Improper tomography operators for tomography specified.')
                print('In \'tomo_approx\' keyword, please use:')
                print('(1) \'full\' , (2) \'fo\' , (3) \'so\'')
                sys.exit()
            for op in operators:
                temp = 'sign{}-{}-{}-{}-{}-{}'.format(
                        str(a),str(b),str(c),str(d),str(n),op)
                Q = GenerateDirectCircuit(
                        QuantStore,
                        _name=temp
                        )
                tomo._pauli_2rdm(Q,a,b,c,d,pauli=op)
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append([temp])
                circuit.append(Q.qc)
            n+=1
    elif QuantStore.tomo_ext=='sign_2e_pauli_symm':
        operators = ['xxxx']
        for a,b,c,d in QuantStore.qc_tomo_quad:
            temp = 'sign{}-{}-{}-{}-{}-{}'.format(
                    str(a),str(b),str(c),str(d),str(n),op)
            Q = GenerateDirectCircuit(
                    QuantStore,
                    _name=temp
                    )
    elif QuantStore.tomo_ext=='sign_2e_from_ancilla':
        operators = ['xxxx','yyyy']
        n=0
        for a,b,c,d in (QuantStore.qc_tomo_quad):
            for op in operators:
                temp = 'sign{}-{}-{}-{}-{}-{}'.format(
                        str(a),str(b),str(c),str(d),str(n),op)
                QuantStore.ec_replace_quad[n]['kw']['pauli']=op
                Q = GenerateDirectCircuit(
                        QuantStore,
                        _name=temp,
                        _flag_sign=True,
                        )
                Q.qc.measure(Q.q,Q.c)
                circuit_list.append([temp])
                circuit.append(Q.qc)
            n+=1 
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
    rawr
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
        QuantStore,
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
            prov.load_accounts()
        try:
            return prov.get_backend(backend)
        except Exception:
            traceback.print_exc()
    beo = _tomo_get_backend(
            QuantStore.provider,
            QuantStore.backend)
    backend_options = {}
    counts = []
    if QuantStore.use_noise:
        noise_model=QuantStore.noise_model
        backend_options['noise_model']=noise_model
        backend_options['basis_gates']=noise_model.basis_gates
        coupling = noise_model.coupling_map
    else:
        if QuantStore.be_file in [None,False]:
            if QuantStore.be_coupling in [None,False]:
                if QuantStore.backend=='qasm_simulator':
                    coupling=None
                else:
                    IBMQ.load_accounts()
                    backend_overview()
                    beo = IBMQ.get_backend(QuantStore.backend)
                    coupling = beo.configuration().coupling_map
            else:
                coupling = QuantStore.be_coupling
        else:
            try:
                coupling = NoiseSimulator.get_coupling_map(
                        device=QuantStore.backend,
                        saved=QuantStore.be_file
                        )
            except Exception as e:
                print(e)
                sys.exit()
    if QuantStore.transpile=='default':
        circuits = transpile(
                circuits=circuits,
                backend=beo,
                coupling_map=coupling,
                initial_layout=QuantStore.be_initial,
                **QuantStore.transpiler_keywords
                )
    else:
        sys.exit('Configure pass manager.')
    qo = assemble(
            circuits,
            shots=QuantStore.Ns
            )
    if QuantStore.use_noise:
        try:
            job = beo.run(qo,backend_options=backend_options)
        except Exception as e:
            traceback.print_exc()
        for circuit in circuit_list:
            name = circuit[0]
            counts.append(job.result().get_counts(name))
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
    return zip(circuit_list,counts)

