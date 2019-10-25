from hqca.quantum.BuildCircuit import GenerateDirectCircuit
from hqca.tools.Fermi import FermiOperator as Fermi
from hqca.tools import Functions as fx
from hqca.tools.RDMFunctions import check_2rdm
from hqca.quantum._ReduceCircuit import simplify_tomography
import numpy as np
from scipy import stats
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
            QuantStore,
            preset_grouping=False,
            mapping=None,
            tomography_terms=None,
            rdm_elements=None,
            **kw):
        self.qs = QuantStore
        self.circuits = []
        self.run = False
        self.Nq = QuantStore.Nq
        self.circuit_list = []
        self.grouping = preset_grouping
        self.mapping = mapping
        self.op = tomography_terms
        self.rdme = rdm_elements
        pass
    
    def construct_rdm(self,**kwargs):
        try:
            self.rdme[0]
        except Exception:
            sys.exit('Have not specified the rdme elements for tomography.')
        try:
            self.counts
        except AttributeError:
            sys.exit('Did you forget to run the circuit? No counts available.')
        self._build_2RDM(**kwargs)

    def _build_2RDM(self,variance=False):
        nRDM = np.zeros((self.Nq,self.Nq,self.Nq,self.Nq),dtype=np.complex_)
        if variance:
            vRDM = np.zeros((self.Nq,self.Nq,self.Nq,self.Nq),dtype=np.complex_)
        for r in self.rdme:
            temp = 0
            tempv = 0
            for Pauli,coeff in zip(r.pauliGates,r.pauliCoeff):
                get = self.mapping[Pauli]
                zMeas = self.__measure_z_string(
                        self.counts[get],
                        Pauli)
                if variance:
                    #p = self._variance_z_string_binomial(
                    #        self.counts[get],Pauli)
                    p = (zMeas+1)/2
                    tempv+= coeff*p*(1-p)
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
                    nRDM[ind1]+=temp*s/2 #factor of 2 is for double counting
                    nRDM[ind2]+=np.conj(temp)*s/2
                    if variance:
                        vRDM[ind1]+=tempv*s/2
                        vRDM[ind2]+=np.conj(tempv)*s/2

        self.rdm2 = RDMs(
                order=2,
                alpha=self.qs.alpha['active'],
                beta=self.qs.beta['active'],
                state='given',
                Ne=self.qs.Ne,
                rdm=nRDM)
        if variance:
            self.rdm2_var = RDMs(
                    order=2,
                    alpha=self.qs.alpha['active'],
                    beta=self.qs.beta['active'],
                    state='given',
                    Ne=self.qs.Ne,
                    rdm=vRDM)

    def _build_mod_2RDM(self,counts):
        nRDM = np.zeros((self.Nq,self.Nq,self.Nq,self.Nq),dtype=np.complex_)
        for r in self.rdme:
            temp = 0
            tempv = 0
            for Pauli,coeff in zip(r.pauliGates,r.pauliCoeff):
                get = self.mapping[Pauli]
                zMeas = self.__measure_z_string(
                        counts[get],
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
                    nRDM[ind1]+=temp*s/2 #factor of 2 is for double counting
                    nRDM[ind2]+=np.conj(temp)*s/2
        rdm =  RDMs(
                order=2,
                alpha=self.qs.alpha['active'],
                beta=self.qs.beta['active'],
                state='given',
                Ne=self.qs.Ne,
                rdm=nRDM)
        return rdm

    def _variance_z_string_binomial(self,counts,zstr):
        val,total= 0,0
        binom = {1:0,-1:0}
        for det,n in counts.items():
            ph=1
            for i,z in enumerate(zstr):
                if z in ['I','i']:
                    continue
                if det[self.Nq-i-1]=='1':
                    ph*=-1
            binom[ph]+=n
            total+= n
        k = binom[1]
        # now, need to estimate ML of binomial distribution
        return k/total

    
    def evaluate_error(
            self,
            numberOfSamples=256, # of times to repeat
            sample_size=4096, # number of counts in sample
            ci=0.90, #target CI#,
            f=None,
            replace=False,
            spin_alt=False
            ):
        count_list = []
        N = self.qs.Ns
        if sample_size>=N:
            sample_size=int(N*0.5)
        samplesSD = []
        sample_means = []
        for t in range(numberOfSamples):
            sample_mean  = f(
                    self.getRandomRDMFromCounts(
                        self.counts,sample_size
                        )
                    )
            sample_means.append(sample_mean)
        t = stats.t.ppf(ci,N)
        std_err = np.std(np.asarray(sample_means),axis=0) #standard error of mean
        ci = std_err*np.sqrt(sample_size/N)*t
        return ci

    def getRandomRDMFromCounts(self,all_counts,length):
        counts_list = {}
        for pauli,counts in all_counts.items():
            count_list = []
            for k,v in counts.items():
                count_list = count_list+[k]*v
            counts_list[pauli]=count_list
        random_counts = {}
        for pauli,clist in counts_list.items():
            random_counts[pauli]={}
            sample_list = np.random.choice(clist,length,replace=False)
            for j in sample_list:
                try:
                    random_counts[pauli][j]+=1
                except KeyError:
                    random_counts[pauli][j]=1
        return self._build_mod_2RDM(random_counts)

    def generate_2rdme(self,real=True,imag=False):
        if not self.grouping:
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

    def _generate_pauli_measurements(self):
        paulis = []
        for fermi in self.rdme:
            for j in fermi.pauliGates:
                if j in paulis:
                    pass
                else:
                    paulis.append(j)
        self.op,self.mapping = simplify_tomography(paulis)


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
