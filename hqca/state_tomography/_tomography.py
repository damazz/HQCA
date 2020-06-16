import numpy as np
from functools import reduce,partial
from scipy import stats
from copy import deepcopy as copy
import sys
import traceback
from timeit import default_timer as dt
from hqca.core import *
from hqca.tools import *
from hqca.circuits import *
from hqca.state_tomography._reduce_circuit import simplify_tomography
from hqca.state_tomography._reduce_circuit import compare_tomography
from hqca.processes import *
from hqca.core.primitives import *
from qiskit import transpile,assemble,execute

class RDMElement:
    def __init__(self,op,transform,ind=None,**kw):
        self.rdmOp = op
        self.qubOp = op.transform(transform)
        try:
            ind[0]
            self.ind = ind
        except Exception:
            self.ind = op.op[0].inds()

class StandardTomography(Tomography):
    '''
    Standard Tomography with optional
    '''
    def __init__(self,
            QuantStore,
            preset=False,
            verbose=True,
            Nq=None,
            dim=None,
            order=None,
            **kw):
        self.grouping = False
        self.run = False
        if type(QuantStore)==type(None):
            self.Nq= Nq
            self.Nq_tot = Nq
            self.p = order
        else:
            self.Nq = QuantStore.Nq
            self.Nq_tot = QuantStore.Nq_tot
            self.qs = QuantStore
            self.p = QuantStore.p
        if preset:
            self._preset_configuration(**kw)
        self.dim = tuple([
            self.qs.dim for i in range(2*self.p)])
        self.circuits = []
        self.circuit_list = []
        self.verbose=verbose
        self.op_type = QuantStore.op_type

    def _preset_configuration(self,
            Tomo=None,
            ):
        #self.grouping = Tomo.preset_cliques
        self.grouping=True
        self.mapping = Tomo.mapping
        self.op = Tomo.op
        #print(self.grouping)
        self.rdme = Tomo.rdme
        self.real = Tomo.real
        self.imag = Tomo.imag

    def set(self,Instruct):
        if self.verbose:
            print('Generating circuits to run.')
        i=0
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenericCircuit(
                    QuantStore=self.qs,
                    _name=circ,
                    )
            init = Operator()
            init+= PauliString('I'*self.qs.Nq,1)
            for n,item in enumerate(self.qs.initial):
                tem = Operator()
                op1 = FermiString(1,
                        indices=[item],
                        ops='+',
                        N=self.qs.dim)
                op2 = FermiString( -1,
                        indices=[item],
                        ops='-',
                        N=self.qs.dim,
                        )
                tem+=op1
                tem+=op2
                try:
                    new = tem.transform(self.qs.initial_transform)
                    init*= new
                except AttributeError:
                    new = tem.transform(self.qs.transform)
                    init*= new
            try:
                U = self.qs.initial_clifford
                apply_clifford_operation(Q,U)
            except AttributeError as e:
                pass
                #print(e)
            except Exception as e:
                print('Error in applying initial clifford transformation.')
                sys.exit(e)
            for s in init:
                apply_pauli_string(Q,s)
            Q.apply(Instruct=Instruct)
            if self.verbose:
                if i==0:
                    print(Q.qc.qasm())
                    i+=1 
            for n,q in enumerate(circ):
                pauliOp(Q,n,q)
            if self.qs.backend in ['unitary_simulator','statevector_simulator']:
                pass
            else:
                Q.qc.measure(Q.q,Q.c)
            self.circuits.append(Q.qc)


    def construct(self,
            **kwargs):
        try:
            self.rdme[0]
        except Exception:
            sys.exit('Have not specified the rdme elements for tomography.')
        try:
            self.counts
        except AttributeError:
            sys.exit('Did you forget to run the circuit? No counts available.')
        if self.op_type=='fermionic':
            self._build_fermionic_RDM(**kwargs)
        elif self.op_type=='qubit':
            self._build_qubitRDM()
            if self.qs.post:
                self._apply_qubit_symmetry(self.qs._symm,**kwargs)

    def _apply_fermionic_symmetry(self,symmetry,rdm=None):
        rdm = copy(rdm)
        rho = RDM(order=2,
                Nq=self.Nq,state='blank')

    def _apply_qubit_symmetry(self,symmetry,rdm=None):
        rdm = copy(rdm)
        if not self.Nq==2:
            sys.exit('No symmetry configured for N=/=2')
        rho = qRDM(order=2,
                Nq=self.Nq,
                state='blank')
        rho = RDM(
                order=self.p,
                alpha=self.qs.groups[0],
                beta=self.qs.groups[1],
                state='blank',
                Ne=self.qs.Ne,
                )
        if not self.real and self.imag:
            rho.rdm = rdm.rdm + self.rdm.rdm
        else:
            rho = self.rdm
        for s in symmetry:
            M = self._get_symmetry_matrix(s)
            norm = rho.observable(np.asarray([M]))
            rho.rdm = np.asarray([
                    reduce(np.dot, (M,rho.rdm[0],M))*1/norm
                    ]
                    )
        if not self.real and self.imag:
            self.rdm.rdm = 1j*np.imag(rho.rdm)
        else:
            self.rdm = rho

    def _get_symmetry_matrix(self,symmetry):
        mat = Circ(self.Nq)
        for n,p in enumerate(symmetry):
            if p=='X':
                mat.x(n)
            elif p=='Y':
                mat.y(n)
            elif p=='Z':
                mat.z(n)
        return mat.m

    def _build_fermionic_RDM(self,
            processor=None,
            variance=False):
        if type(processor)==type(None):
            processor=StandardProcess()
        nRDM = np.zeros(self.dim,dtype=np.complex_)
        for r in self.rdme:
            temp = 0
            for op in r.qubOp:
                get = self.mapping[op.s] #self.mapping has important get
                # property to get the right pauli
                zMeas = processor.process(
                        counts=self.counts[get],
                        pauli_string=op.s,
                        quantstore=self.qs,
                        backend=self.qs.backend,
                        Nq=self.qs.Nq_tot)
                temp+= zMeas*op.c
            opAnn = r.ind[2:][::-1]
            opCre = r.ind[0:2]
            reAnn = Recursive(choices=opAnn)
            reCre = Recursive(choices=opCre)
            reAnn.unordered_permute()
            reCre.unordered_permute()
            for i in reAnn.total:
                for j in reCre.total:
                    ind1 = tuple(j[:self.p]+i[:self.p])
                    s = i[self.p]*j[self.p]
                    nRDM[ind1]+=temp*s #factor of 2 is for double counting
                    if not set(i[:2])==set(j[:2]):
                        ind2 = tuple(i[:self.p]+j[:self.p])
                        nRDM[ind2]+=np.conj(temp)*s
        self.rdm = RDM(
                order=self.p,
                alpha=self.qs.groups[0],
                beta=self.qs.groups[1],
                state='given',
                Ne=self.qs.Ne,
                rdm=nRDM)

    def _build_qubitRDM(self):
        if self.p==1:
            self._build_qubit1RDM()
        else:
            self._build_qubit2RDM()

    def _build_qubit1RDM(self):
        self.rdm = qRDM(
                order=self.p,
                Nq=self.Nq,
                state='blank',)
        for r in self.rdme:
            temp=0
            for Pauli,coeff in zip(r.pPauli,r.pCoeff):
                try:
                    get = self.mapping[Pauli] #self.mapping has important get
                    # property to get the right pauli
                    zMeas = self.__measure_z_string(
                            self.counts[get],
                            Pauli)
                    temp+= zMeas*coeff
                except KeyError as e:
                    pass
                    #print('Key not found')
            if r.sqOp=='p':
                self.rdm.rdm[0,1,1]+=temp
            elif r.sqOp=='h':
                self.rdm.rdm[0,0,0]+=temp
            elif r.sqOp=='-':
                self.rdm.rdm[0,0,1]+=temp
            elif r.sqOp=='+':
                self.rdm.rdm[0,1,0]+=temp

    def _build_qubit2RDM(self,processor=None,variance=False):
        if type(processor)==type(None):
            processor=StandardProcess()
        self.rdm = qRDM(order=2,
                Nq=self.Nq,
                state='blank')
        for r in self.rdme:
            temp = 0
            for Pauli,coeff in zip(r.pPauli,r.pCoeff):
                try:
                    get = self.mapping[Pauli] #self.mapping has important get
                    # property to get the right pauli
                    zMeas = processor.process(
                            counts=self.counts[get],
                            pauli_string=Pauli,
                            quantstore=self.qs,
                            backend=self.qs.backend,
                            Nq=self.qs.Nq_tot)
                    #zMeas = self.__measure_z_string(
                    #        self.counts[get],
                    #        Pauli)
                    temp+= zMeas*coeff
                except KeyError as e:
                    pass
            ia = self.rdm.rev_map[tuple(r.qInd)]
            ib = self.rdm.rev_sq[r.sqOp]
            ind = tuple([ia])+ib
            self.rdm.rdm[ind]+= temp

    def generate(self,**kw):
        if self.op_type=='fermionic':
            if self.p==2:
                self._generate_2rdme(**kw)
            elif self.p==1:
                self._generate_1rdme(**kw)
        elif self.op_type=='qubit':
            if self.p==2:
                self._generate_2qrdme(**kw)
            elif self.p==1:
                self._generate_1qrdme(**kw)
        self._generate_pauli_measurements(**kw)

    def _generate_1qrdme(self,**kw):
        '''
        generates 1-local properties, i.e. local qubit properties
        this includes the set of 
        '''
        rdme = []
        if not self.grouping:
            def sub_rdme(i,op):
                test = QubitOperator(
                        coeff=1,
                        indices=[i],
                        sqOp=op)
                test.generateTomography(Nq=self.Nq,
                        Nq_tot=self.Nq_tot,**kw)
                return test
            for i in range(self.Nq):
                rdme.append(sub_rdme(i,'+'))
                rdme.append(sub_rdme(i,'-'))
                rdme.append(sub_rdme(i,'p'))
                rdme.append(sub_rdme(i,'h'))
        self.rdme = rdme
        self.real,self.imag = True,True

    def _generate_2qrdme(self,real=True,imag=True,**kw):
        '''
        generates 1-local properties, i.e. local qubit properties
        this includes the set of
        '''
        rdme = []
        self.real = real
        self.imag = imag
        if not self.grouping:
            def sub_rdme(i,j,op):
                test = QubitOperator(
                        coeff=1,
                        indices=[i,j],
                        sqOp=op)
                test.generateTomography(
                        Nq=self.Nq,
                        Nq_tot=self.Nq_tot,
                        real=real,
                        imag=imag,
                        **kw)
                return test
            for j in range(self.Nq):
                for i in range(j):
                    for p in ['+','-','p','h']:
                        for k in ['+','-','p','h']:
                            rdme.append(sub_rdme(i,j,p+k))
        self.rdme = rdme

    def _generate_1rdme(self,real=True,imag=False,**kw):
        self.real=real
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []
            def sub_rdme(i,j,spin):
                test = FermiString(
                    coeff=1,
                    indices=[i,j],
                    sqOp='+-',
                    spin=spin)
                test.generateTomography(Nq=self.Nq,
                        Nq_tot=self.Nq_tot,
                        real=real,
                        imag=imag,
                        **kw)
                return test
            for i in alp:
                for j in alp:
                    if i*Na>j*Na:
                        continue
                    if imag and i*Na==j*Na:
                        continue
                    new = sub_rdme(i,j,'aa')
                    rdme.append(new)
            for i in bet:
                for j in bet:
                    if i*Na>j*Na:
                        continue
                    if imag and i*Na==j*Na:
                        continue
                    new = sub_rdme(i,j,'bb')
                    rdme.append(new)
            self.rdme = rdme

    def _generate_2rdme(self,real=True,imag=False,verbose=False,
            **kw):
        self.real=real
        kw['verbose']=verbose
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []

            def sub_rdme(i,k,l,j,**kw):
                op = Operator()
                if self.real and self.imag:
                    c1,c2=1,0
                elif self.real and not self.imag:
                    c1,c2=0.5,0.5
                elif not self.real and self.imag:
                    c1,c2 = 0.5,-0.5
                test = FermiString(
                    coeff=c1,
                    indices=[i,k,l,j],
                    ops='++--',
                    N=self.qs.dim,
                    )
                op+=test
                test = FermiString(
                    coeff=c2,
                    indices=[j,l,k,i],
                    ops='++--',
                    N=self.qs.dim,
                    )
                op+= test
                return RDMElement(op,ind=[i,k,l,j],**kw)
            if verbose:
                print('Generating alpha-alpha block of 2-RDM')
            for i in alp:
                for k in alp:
                    if i>=k:
                        continue
                    for l in alp:
                        for j in alp:
                            if j>=l or i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            new= sub_rdme(i,k,l,j,**kw)
                            rdme.append(new)
            if verbose:
                print('Generating beta-beta block of 2-RDM')
            for i in bet:
                for k in bet:
                    if i>=k:
                        continue
                    for l in bet:
                        for j in bet:
                            if j>=l or i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            new = sub_rdme(i,k,l,j,**kw)
                            rdme.append(new)
            if verbose:
                print('Generating alpha-beta block of 2-RDM')
            for i in alp:
                for k in bet:
                    for l in bet:
                        for j in alp:
                            if i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            new = sub_rdme(i,k,l,j,**kw)
                            rdme.append(new)
            self.rdme = rdme

    def _pauli_commutation(self,L,R):
        new = ''
        for a,b in zip(L,R):
            if a=='I':
                s=b[:]
            elif b=='I':
                s=a[:]
            elif a==b:
                s='I'
            else:
                p = set(['X','Y','Z'])
                p.remove(a)
                p.remove(b)
                s = p.pop()
            new = new+s
        return new

    def _generate_pauli_measurements(self,
            simplify=True,
            symmetries=[],
            **kw):
        paulis = []
        for fermi in self.rdme:
            for j in fermi.qubOp:
                if j.s in paulis:
                    pass
                else:
                    paulis.append(j.s)
        if simplify==True:
            self.op,self.mapping = simplify_tomography(
                    paulis,
                    **kw)
        elif simplify=='comparison':
            self.op,self.mapping = compare_tomography(
                    paulis,
                    **kw)
        else:
            self.op = paulis
            self.mapping = {p:p for p in paulis}

    def _transform_q2r(self,rdme):
        nrdme = []
        for i in rdme:
            nrdme.append(self.qs.qubit_to_rdm[i])
        return nrdme


    def simulate(self,verbose=False):
        beo = self.qs.beo
        backend_options = {}
        counts = []
        if self.qs.use_noise:
            backend_options['noise_model']=self.qs.noise_model
            backend_options['basis_gates']=self.qs.noise_model.basis_gates
            coupling = self.qs.noise_model.coupling_map
        else:
            if self.qs.be_file in [None,False]:
                if self.qs.be_coupling in [None,False]:
                    if self.qs.backend=='qasm_simulator':
                        coupling=None
                    else:
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
            #print(circuits[0])
        else:
            sys.exit('Configure pass manager.')
        qo = assemble(
                circuits,
                shots=self.qs.Ns
                )
        if self.qs.backend=='unitary_simulator':
            job = beo.run(qo)
            for circuit in self.circuit_list:
                counts.append(job.result().get_counts(name))
        elif self.qs.backend=='statevector_simulator':
            job = beo.run(qo)
            for n,circuit in enumerate(self.circuit_list):
                #if self.verbose and self.Nq<=4:
                #    print('Circuit: {}'.format(circuit))
                #    print(job.result().get_statevector(circuit))
                counts.append(job.result().get_statevector(circuit))
        elif self.qs.use_noise:
            try:
                job = beo.run(
                        qo,
                        backend_options=backend_options,
                        noise_model=self.qs.noise_model,
                        )
            except Exception as e:
                traceback.print_exc()
            for circuit in self.circuit_list:
                name = circuit
                counts.append(job.result().get_counts(name))
        else:
            try:
                job = beo.run(qo)
                for circuit in self.circuit_list:
                    name = circuit
                    counts.append(job.result().get_counts(name))
                for circ in self.circuits:
                    if circ.name=='Z':
                        print(circ)
                    elif circ.name=='ZZ':
                        print(circ)
                    elif circ.name=='XY':
                        print(circ)
            except Exception as e:
                print('Error: ')
                print(e)
                traceback.print_exc()
        if self.qs.use_meas_filter:
            self.counts  = {}
            for i,j in zip(self.circuit_list,counts):
                c = self.qs.meas_filter.apply(
                    j,
                    method='least_squares'
                    )
                self.counts[i]=c
        else:
            self.counts = {i:j for i,j in zip(self.circuit_list,counts)}
        if self.verbose:
            for i,j in self.counts.items():
                if self.qs.backend=='statevector_simulator':
                    pass
                else:
                    print(i,j)

    def evaluate_error(
            self,
            numberOfSamples=256, # of times to repeat
            sample_size=1024, # number of counts in sample
            ci=0.90, #target CI#,
            f=None,
            replace=False,
            spin_alt=False
            ):
        print('Samples: {}'.format(numberOfSamples))
        print('Sample size: {}'.format(sample_size))
        count_list = []
        N = self.qs.Ns
        if sample_size>=N*8:
            sample_size=int(N/8)
        samplesSD = []
        sample_means = []
        counts_list = {}
        for pauli,counts in self.counts.items():
            count_list = []
            for k,v in counts.items():
                count_list = count_list+[k]*v
            counts_list[pauli]=count_list
        for t in range(numberOfSamples):
            t1 = dt()
            sample_mean  = f(
                    self.getRandomRDMFromCounts(
                        counts_list,sample_size
                        )
                    )
            if np.isnan(sample_mean):
                continue
            else:
                sample_means.append(sample_mean)
            t2 = dt()
            #print('Time: {}'.format(t2-t1))
        t = stats.t.ppf(ci,N)
        std_err = np.std(np.asarray(sample_means),axis=0) #standard error of mean
        ci = std_err*np.sqrt(sample_size/N)*t
        return ci

    def getRandomRDMFromCounts(self,counts_list,length):
        random_counts = {}
        for pauli,clist in counts_list.items():
            random_counts[pauli]={}
            sample_list = np.random.choice(clist,length,replace=False)
            for j in sample_list:
                try:
                    random_counts[pauli][j]+=1
                except KeyError:
                    random_counts[pauli][j]=1
        #print('Build random list: {}'.format(t3-t5))
        del self.rdm
        self.counts = random_counts
        self.construct()
        #new = self._build_mod_2RDM(random_counts)
        #print('Build 2rdm: {}'.format(t4-t3))
        return self.rdm
