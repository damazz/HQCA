from hqca.quantum.BuildCircuit import GenerateDirectCircuit
from hqca.tools.Fermi import FermiOperator as Fermi
from hqca.tools import Functions as fx
from hqca.tools.RDMFunctions import check_2rdm
import numpy as np
from hqca.quantum.BuildCircuit import GenerateCircuit
from hqca.quantum.primitives import _Tomo as tomo
from qiskit import Aer,IBMQ,execute
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from qiskit.tools.monitor import backend_overview,job_monitor
from hqca.tools.RDM import Recursive,RDMs
import sys

class Tomography:
    def __init__(self,
            QuantStore):
        self.qs = QuantStore
        self.circuits = []
        self.run = False
        self.Nq = QuantStore.Nq
        self.circuit_list = []
        self.imTomo = False
        pass
    
    def construct_rdm(self):
        try:
            self.rdme[0]
        except Exception:
            sys.exit('Have not specified the rdme elements for tomography.')
        self._build_2RDM()

    def _build_2RDM(self):
        nRDM = np.zeros((self.Nq,self.Nq,self.Nq,self.Nq),dtype=np.complex_)
        #print(self.counts['ZZZZ'])
        for r in self.rdme:
            temp = 0
            for get,Pauli,coeff in zip(r.pauliGet,r.pauliGates,r.pauliCoeff):
                zMeas = self.__measure_z_string(
                        self.counts[get],
                        Pauli)
                temp+= zMeas*coeff
            opAnn = r.ind[2:][::-1]
            opCre = r.ind[0:2]
            reAnn = Recursive(choices=opAnn)
            reCre = Recursive(choices=opCre)
            reAnn.unordered_permute()
            reCre.unordered_permute()
            for i in reAnn.total:
                for j in reCre.total:
                    ind1 = tuple(j[:2]+i[:2])
                    ind2 = tuple(i[:2]+j[:2])
                    s = i[2]*j[2]
                    nRDM[ind1]+=temp*s#*0.5
                    if set(opAnn).difference(set(opCre)):
                        nRDM[ind2]+=np.conj(temp)*s#*0.5
        self.rdm2 = RDMs(
                order=2,
                alpha=self.qs.alpha['active'],
                beta=self.qs.beta['active'],
                state='given',
                Ne=self.qs.Ne,
                rdm=nRDM)


    def generate_2rdme(self,real=True,imag='default'):
        if imag=='default':
            imag=self.imTomo
        alpha = self.qs.alpha['active']
        rdme = []
        beta = self.qs.beta['active']
        S = []
        blocks = [
                [alpha,alpha,beta],
                [alpha,beta,beta],
                [alpha,beta,beta],
                [alpha,alpha,beta]
                ]
        block = ['aa','ab','bb']
        for ze in range(len(blocks[0])):
            for i in blocks[0][ze]:
                for k in blocks[1][ze]:
                    for l in blocks[2][ze]:
                        for j in blocks[3][ze]:
                            if block[ze]=='ab':
                                if i>j or k>l:
                                    continue
                                spin = ['abba']
                            else:
                                if i>=k or j>=l:
                                    continue
                                if block[ze]=='aa':
                                    spin = ['aaaa']
                                else:
                                    spin = ['bbbb']
                            test = Fermi(
                                coeff=1,
                                indices=[i,k,l,j],
                                sqOp='++--',
                                spin=spin[0])
                            test.generateTomoBasis(
                                    real=real,
                                    imag=imag,
                                    Nq=self.qs.Nq)
                            rdme.append(test)
        self.rdme = rdme
        self.generate_pauli_measurements()

    def generate_pauli_measurements(self):
        self.paulis = []
        for fermi in self.rdme:
            for j in fermi.pauliGet:
                if j in self.paulis:
                    pass
                else:
                    print(j)
                    self.paulis.append(j)
        self.op = self.paulis

    def _transform_q2r(self,rdme):
        nrdme = []
        for i in rdme:
            nrdme.append(self.qs.qubit_to_rdm[i])
        return nrdme

    def __measure_z_string(self,counts,zstr):
        val,total= 0,0
        for det,n in counts.items():
            ph=1
            for i,z in enumerate(zstr):
                if z in ['I','i']:
                    continue
                if det[self.Nq-i-1]=='1':
                    ph*=-1
            val+= n*ph
            total+=n
        return val/total

    def _gen_pauli_str(self,st):
        '''
        from a list, generates the corresponding pauli string
        '''
        i,j,k,l,sp,sq = st[0],st[1],st[2],st[3],st[4],st[5]
        if self.imTomo==False:
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
                if temp in self.op:
                    pass
                else:
                    self.op.append(temp)
        else:
            ops = ['xxxx','xxyy','yxxx','xyxx']
            if sp in ['abba','baab']:
                ops = ['xxxx','xyyx','xxxy','yxxx']
            elif sp in ['abab','baba']:
                ops = ['xxxx','xyxy','xxyx','yxxx']
            elif sp in ['aabb','bbaa']:
                ops = ['xxxx','xxyy','yxxx','xyxx']
            signs = [1/4,1/4,1j/4,-1j/4]
            for op in ops:
                temp = 'i'*self.Nq
                temp=temp[:i]+op[0]+temp[i+1:]
                temp=temp[:j]+op[1]+temp[j+1:]
                temp=temp[:k]+op[2]+temp[k+1:]
                temp=temp[:l]+op[3]+temp[l+1:]
                if temp in self.op:
                    pass
                else:
                    self.op.append(temp)

    def run_circuit(self):
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
