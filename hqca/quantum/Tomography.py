from hqca.quantum.BuildCircuit import GenerateDirectCircuit
from hqca.tools.Fermi import FermiOperator as Fermi
from hqca.tools import Functions as fx
from hqca.tools.RDMFunctions import check_2rdm
from hqca.quantum._ReduceCircuit import simplify_tomography
import numpy as np
from hqca.quantum.BuildCircuit import GenerateCircuit
from hqca.quantum.primitives import _Tomo as tomo
from qiskit import Aer,IBMQ,execute
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from qiskit.tools.monitor import backend_overview,job_monitor
from hqca.tools.RDM import Recursive,RDMs
import sys

#
#class SetTomography(Tomography):
#    def __init__(self,**kwargs):
#        Tomography.__init__(self,**kwargs)
#        pass


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
        try:
            self.counts
        except AttributeError:
            sys.exit('Did you forget to run the circuit? No counts available.')
        self._build_2RDM()

    def _build_2RDM(self):
        nRDM = np.zeros((self.Nq,self.Nq,self.Nq,self.Nq),dtype=np.complex_)
        for r in self.rdme:
            #print(r.ind)
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
                    nRDM[ind1]+=temp*s/2
                    nRDM[ind2]+=np.conj(temp)*s/2
        self.rdm2 = RDMs(
                order=2,
                alpha=self.qs.alpha['active'],
                beta=self.qs.beta['active'],
                state='given',
                Ne=self.qs.Ne,
                rdm=nRDM)


    def generate_2rdme(self,real=True,imag=False):
        alp = self.qs.alpha['active']
        Na = len(alp)
        rdme = []
        bet = self.qs.beta['active']
        S = []

        def sub_rdme(i,k,l,j,spin):
            test = Fermi(
                coeff=1,
                indices=[i,k,l,j],
                sqOp='++--',
                spin=spin)
            test.generateTomoBasis(
                    real=real,
                    imag=imag,
                    Nq=self.qs.Nq)
            return test
        for i in alp:
            for k in alp:
                if i>=k:
                    continue
                for l in alp:
                    for j in alp:
                        if j>=l or i*Na+k>j*Na+l:
                            continue
                        if imag and i*Na+k==j*Na+l:
                            continue
                        new = sub_rdme(i,k,l,j,'aaaa')
                        rdme.append(new)
        for i in bet:
            for k in bet:
                if i>=k:
                    continue
                for l in bet:
                    for j in bet:
                        if j>=l or i*Na+k>j*Na+l:
                            continue
                        if imag and i*Na+k==j*Na+l:
                            continue
                        new = sub_rdme(i,k,l,j,'bbbb')
                        rdme.append(new)
    
        for i in alp:
            for k in bet:
                for l in bet:
                    for j in alp:
                        if i*Na+k>j*Na+l:
                            continue
                        if imag and i*Na+k==j*Na+l:
                            continue
                        new = sub_rdme(i,k,l,j,'abba')
                        rdme.append(new)
        self.rdme = rdme
        self._generate_pauli_measurements()

    def _use_reduced_setting(self):
        self.rdme = simplify_tomography(self.rdme)
        pass

    def _generate_pauli_measurements(self):
        self.paulis = []
        #self._use_reduced_setting()
        for fermi in self.rdme:
            for j in fermi.pauliGet:
                if j in self.paulis:
                    pass
                else:
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


    def run_circuit(self,verbose=False):
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
