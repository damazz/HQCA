from hqca.quantum.BuildCircuit import GenerateDirectCircuit
import numpy as np
from hqca.quantum.BuildCircuit import GenerateCircuit
from hqca.quantum.algorithms import _Tomo as tomo
from qiskit import Aer,IBMQ,execute
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from qiskit.tools.monitor import backend_overview,job_monitor
from hqca.tools.RDM import Recursive,RDMs

class Tomography:
    def __init__(self,
            QuantStore):
        self.qs = QuantStore
        self.circuits = []
        self.run = False
        self.Nq = QuantStore.Nq
        self.circuit_list = []
        pass

    def build_circuit(self):
        if self.qs.tomo_rdm=='1rdm':
            pass
        elif self.qs.tomo_rdm=='2rdm':
            pass
        elif self.qs.tomo_rdm=='3rdm':
            pass
        elif self.qs.tomo_rdm=='acse':
            self._gen_acse_ansatz()


    def _gen_acse_ansatz(self):
        self._transform_S()
        self._gen_tomo_list()
        # simplify? 
        self._gen_quantum_S()

    def _gen_quantum_S(self):
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenerateDirectCircuit(self.qs,_name=circ)
            for n,q in enumerate(circ):
                tomo._apply_pauli_op(Q,n,q)
            Q.qc.measure(Q.q,Q.c)
            self.circuits.append(Q.qc)

    def _build_rdms(self): 
        nRDM = np.zeros((self.Nq,self.Nq,self.Nq,self.Nq))
        for r in self.rdme:
            get,pauli,term = self._get_paulis(r)
            for g,p,t in zip(get,pauli,term):
                z = self.__measure_z_string(
                        self.counts[g],
                        p)
                print(r)
                #nRDM[list(r



    def _get_paulis(self,rdme):
        # self.approx 
        unique = set(rdme)
        if len(unique)==2:
            op = 'i'*self.Nq
            u = list(unique)
            op1 = op[:u[0]]+'z'+op[u[0]+1:]
            op2 = op[:u[1]]+'z'+op[u[1]+1:]
            op3 = op[:u[0]]+'z'+op[u[0]+1:]
            op3 = op3[:u[1]]+'z'+op3[u[1]+1:]
            ops = [op,op1,op2,op3]
            get = ['z'*self.Nq]*4
            sign = [1,-1,-1,1]
        elif len(unique)==4:
            ops = ['xxxx','xxyy']
            get = ['xxxx','xxyy']
            sign = [1/4,1/4]
        return get,ops,sign

    
    def __measure_z_string(self,counts,zstr):
        val,total= 0,0
        for det,n in counts.items():
            print(det)
            ph=1
            for i,z in enumerate(zstr):
                if z=='i':
                    continue
                if det[self.Nq-i-1]=='1':
                    ph*=-1
            val+= n*ph
            total+=n
        return val/total
    
    def _gen_tomo_list(self):
        self.op = []
        self.op.append('z'*self.Nq)
        for i in self.S:
            self.__gen_pauli_str(i)

    def __gen_pauli_str(self,st):
        i,j,k,l,sp,sq = st[0],st[1],st[2],st[3],st[4],st[5]
        if sp in ['abba','baab']:
            ops = ['xxxx','xyyx']
        elif sp in ['abab','baba']:
            ops = ['xxxx','xyxy']
        elif sp in ['aabb','bbaa']:
            ops = ['xxxx','xxyy']
        signs = [1/4,1/4]
        ops = ['xxxx','xxyy']
        for op in ops:
            temp = 'i'*self.Nq
            temp=temp[:i]+op[0]+temp[i+1:]
            temp=temp[:j]+op[1]+temp[j+1:]
            temp=temp[:k]+op[2]+temp[k+1:]
            temp=temp[:l]+op[3]+temp[l+1:]
            self.op.append(temp)


    def _transform_S(self):
        '''
        put S into qubit language
        '''
        rdme = []
        S = []
        para = []
        for qd in self.qs.S:
            #for i in range(len(quad)-2):
            #    qd.append(self.rdm_to_qubit[quad[i]])
            sort = False
            try:
                sign = list(qd[5])
                spin = list(qd[6])
            except IndexError:
                sign = ['+','+','-','-']
                spin = ['a','b','b','a']
            while not sort:
                sort = True
                if qd[0]>qd[1]:
                    qd[1],qd[0]=qd[0],qd[1]
                    sign[0],sign[1]=sign[1],sign[0]
                    spin[0],spin[1]=spin[1],spin[0]
                    qd[4]*=-1
                    sort = False
                if qd[1]>qd[2]:
                    qd[1],qd[2]=qd[2],qd[1]
                    sign[2],sign[1]=sign[1],sign[2]
                    spin[2],spin[1]=spin[1],spin[2]
                    qd[4]*=-1
                    sort = False
                if qd[2]>qd[3]:
                    sign[3],sign[2]=sign[2],sign[3]
                    spin[2],spin[3]=spin[3],spin[2]
                    qd[3],qd[2]=qd[2],qd[3]
                    qd[4]*=-1
                    sort = False
            qd[5]=''.join(sign)
            qd[6]=''.join(spin)
            para.append(qd.pop(4))
            S.append(qd)
            rdme.append(qd[0:4])
        self.qs.parameters=para
        self.qs.qc_quad_list = S
        self.S= S
        # getting 2rdm measurements
        for a1 in self.qs.alpha['active']:
            for a2 in self.qs.alpha['active']:
                if a1<a2 and a2<self.qs.Ne_alp:
                    rdme.append([a1,a2,a2,a1])
        for a1 in self.qs.alpha['active']:
            for b1 in self.qs.beta['active']:
                if a1<self.qs.Ne_alp and b1<self.qs.Ne_bet+self.qs.No:
                    rdme.append([a1,b1,b1,a1])
        for b1 in self.qs.beta['active']:
            for b2 in self.qs.beta['active']:
                if b1<b2 and b2<self.qs.Ne_bet+self.qs.No:
                    rdme.append([b1,b2,b2,b1])
        new = [ ]
        for m,s1 in enumerate(rdme):
            for n,s2 in enumerate(rdme):
                if m<n:
                    common = set(s1[:4]).intersection(set(s2[:4]))
                    if len(common)==2:
                        diff = list(set(s1[:4]).difference(set(s2[:4])))
                        new.append(diff+diff[::-1])
                    else:
                        print('Error in RDMTomography')
                        sys.exit('Need to configure for these terms!')
        rdme+= new
        self.rdme = rdme



    def run_circuits(self):
        def _tomo_get_backend(
                provider,
                backend
                ):
            if provider=='Aer':
                prov=Aer
            elif provider=='IBMQ':
                prov=IBMQ
                #prov.load_accounts()
            try:
                return prov.get_backend(backend)
            except Exception:
                traceback.print_exc()
        beo = _tomo_get_backend(
                self.qs.provider,
                self.qs.backend)
        backend_options = {}
        counts = []
        if self.qs.use_noise:
            noise_model=self.qs.noise_model
            backend_options['noise_model']=noise_model
            backend_options['basis_gates']=noise_model.basis_gates
            coupling = noise_model.coupling_map
        else:
            if self.qs.be_file in [None,False]:
                if self.qs.be_coupling in [None,False]:
                    if self.qs.backend=='qasm_simulator':
                        coupling=None
                    else:
                        beo = IBMQ.get_backend(self.qs.backend)
                        coupling = beo.configuration().coupling_map
                else:
                    coupling = self.qs.be_coupling
            else:
                try:
                    coupling = NoiseSimulator.get_coupling_map(
                            device=self.qs.backend,
                            saved=self.qs.be_file
                            )
                except Exception as e:
                    print(e)
                    sys.exit()
        if self.qs.transpile=='default':
            circuits = transpile(
                    circuits=self.circuits,
                    backend=beo,
                    coupling_map=coupling,
                    initial_layout=self.qs.be_initial,
                    **self.qs.transpiler_keywords
                    )
        else:
            sys.exit('Configure pass manager.')
        qo = assemble(
                circuits,
                shots=self.qs.Ns
                )
        if self.qs.use_noise:
            try:
                job = beo.run(
                        qo,
                        backend_options=backend_options,
                        noise_model=noise_model
                        )
            except Exception as e:
                traceback.print_exc()
            for circuit in self.circuit_list:
                name = circuit[0]
                counts.append(job.result().get_counts(name))
        else:
            try:
                job = beo.run(qo)
                for circuit in self.circuit_list:
                    name = circuit
                    counts.append(job.result().get_counts(name))
            except Exception as e:
                print('Error: ')
                print(e)
                traceback.print_exc()
        self.counts = {i:j for i,j in zip(self.circuit_list,counts)}


