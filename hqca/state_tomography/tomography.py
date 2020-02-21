import numpy as np
from functools import reduce
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
from hqca.state_tomography._simplify import *
from hqca.core.primitives import *
from qiskit import transpile,assemble,execute

class StandardTomography(Tomography):
    '''
    Standard Tomography with optional 
    '''
    def __init__(self,
            QuantStore,
            preset=False,
            verbose=True,
            match_aa_bb=False,
            **kw):
        self.grouping = False
        if preset:
            self._preset_configuration(**kw)
        self.run = False
        self.Nq = QuantStore.Nq
        self.match_aa_bb = match_aa_bb
        self.qs = QuantStore
        self.p = QuantStore.p
        if self.qs.mapping in ['bk','bravyi-kitaev']:
            if self.qs._kw_mapping['MapSet'].reduced:
                self.dim = tuple([self.Nq+2 for i in range(2*self.p)])
            else:
                self.dim = tuple([self.Nq for i in range(2*self.p)])
        else:
            self.dim = tuple([self.Nq for i in range(2*self.p)])
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
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenericCircuit(
                    QuantStore=self.qs,
                    _name=circ,
                    )
            for item in self.qs.initial:
                if self.qs.mapping in ['jordan-wigner','jw']:
                    Q.qc.x(item)
                elif self.qs.mapping in ['parity']:
                    sys.exit('Need initialization for parity')
                elif self.qs.mapping in ['bk','bravyi-kitaev']:
                    MapSet = self.qs._kw_mapping['MapSet']
                    if MapSet.reduced:
                        if item in MapSet._shifted:
                            Q.qc.x(item-1)
                        else:
                            Q.qc.x(item)
                        for q in MapSet.update[item]:
                            if (q in MapSet._shifted):
                                Q.qc.x(q-1)
                            elif q in MapSet._reduced_set:
                                pass
                            else:
                                Q.qc.x(q)
                    else:
                        Q.qc.x(item)
                        for q in MapSet.update[item]:
                            Q.qc.x(q)
            Q.apply(Instruct=Instruct)
            for n,q in enumerate(circ):
                pauliOp(Q,n,q)
            if self.qs.backend in ['unitary_simulator','statevector_simulator']:
                pass
            else:
                Q.qc.measure(Q.q,Q.c)
            self.circuits.append(Q.qc)

    def construct(self,**kwargs):
        try:
            self.rdme[0]
        except Exception:
            sys.exit('Have not specified the rdme elements for tomography.')
        try:
            self.counts
        except AttributeError:
            sys.exit('Did you forget to run the circuit? No counts available.')
        if self.op_type=='fermionic':
            self._build_fermionic_RDM()
        elif self.op_type=='qubit':
            self._build_qubitRDM()
        if self.qs.post:
            self._apply_symmetry(self.qs._symm,**kwargs)

    def _apply_symmetry(self,symmetry,rdm=None):
        rdm = copy(rdm)
        if not self.Nq==2:
            sys.exit('No symmetry configured for N=/=2')
        rho = qRDM(order=2,
                Nq=self.Nq,
                state='blank')
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

    def _build_fermionic_RDM(self,variance=False):
        nRDM = np.zeros(self.dim,dtype=np.complex_)
        for r in self.rdme:
            temp = 0
            for Pauli,coeff in zip(r.pPauli,r.pCoeff):
                #print(Pauli,coeff)
                get = self.mapping[Pauli] #self.mapping has important get
                # property to get the right pauli
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
                    ind1 = tuple(j[:self.p]+i[:self.p])
                    s = i[self.p]*j[self.p]
                    nRDM[ind1]+=temp*s #factor of 2 is for double counting
                    if not set(i[:2])==set(j[:2]):
                        ind2 = tuple(i[:self.p]+j[:self.p])
                        nRDM[ind2]+=np.conj(temp)*s
        if self.match_aa_bb:
            alp = self.qs.groups[0]
            for i in alp:
                I = self.qs.a2b[i]
                for k in alp:
                    K = self.qs.a2b[k]
                    for l in alp:
                        L = self.qs.a2b[l]
                        for j in alp:
                            J = self.qs.a2b[j]
                            Ind = tuple(I,K,L,J)
                            ind = tuple(i,k,l,j)
                            nrdm[Ind]=nrdm[ind]
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

    def _build_qubit2RDM(self):
        self.rdm = qRDM(order=2,
                Nq=self.Nq,
                state='blank')
        for r in self.rdme:
            temp = 0
            for Pauli,coeff in zip(r.pPauli,r.pCoeff):
                #if r.qOp in ['hh','hp','ph','pp']:
                #    print(r,Pauli,coeff)
                try:
                    get = self.mapping[Pauli] #self.mapping has important get
                    # property to get the right pauli
                    zMeas = self.__measure_z_string(
                            self.counts[get],
                            Pauli)
                    temp+= zMeas*coeff
                except KeyError as e:
                    pass
                    #print('No key for {}->{}'.format(get,Pauli))
            ia = self.rdm.rev_map[tuple(r.qInd)]

            ib = self.rdm.rev_sq[r.sqOp]
            ind = tuple([ia])+ib
            self.rdm.rdm[ind]+= temp


    def generate(self,simplify=True,**kw):
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
        self._generate_pauli_measurements(simplify,**kw)

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
                test.generateTomography(Nq=self.Nq,**kw)
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

    def _generate_2rdme(self,real=True,imag=False,**kw):
        self.real=real
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []
            def sub_rdme(i,k,l,j,spin):
                test = FermionicOperator(
                    coeff=1,
                    indices=[i,k,l,j],
                    sqOp='++--',
                    spin=spin)
                test.generateTomography(Nq=self.Nq,
                        real=real,
                        imag=imag,
                        **kw)
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

    def _generate_pauli_measurements(self,simplify=False,**kw):
        paulis = []
        for fermi in self.rdme:
            for j in fermi.pPauli:
                if j in paulis:
                    pass
                else:
                    paulis.append(j)
        if simplify:
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

    def __measure_z_string(self,counts,zstr):
        if self.qs.backend in ['statevector_simulator']:
            val = 0
            N = 2**self.qs.Nq_tot
            test = ['{:0{}b}'.format(
                i,self.qs.Nq_tot)[::1] for i in range(0,N)]
            for n,b in enumerate(test):
                if abs(counts[n])<1e-14:
                    continue
                sgn = 1
                for i in range(len(b)):
                    if zstr[i]=='I':
                        pass
                    else:
                        if b[self.Nq-i-1]=='1':
                            sgn*=-1
                val+= np.real(counts[n]*np.conj(counts[n])*sgn)
        else:
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
            val = val/total
        return val

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
        else:
            sys.exit('Configure pass manager.')
        qo = assemble(
                circuits,
                shots=self.qs.Ns
                )
        if self.qs.backend=='unitary_simulator':
            job = beo.run(qo)
            for circuit in self.circuit_list:
                #print(job.result().get_unitary(circuit))
                counts.append(job.result().get_counts(name))
        elif self.qs.backend=='statevector_simulator':
            job = beo.run(qo)
            for n,circuit in enumerate(self.circuit_list):
                if verbose:
                    print('Circuit: {}'.format(circuit))
                    print(job.result().get_statevector(circuit))
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


class PseudoRDMElement:
    def __init__(self,pauli_list,ind):
        self.ind = ind
        self.pPauli = []
        self.pCoeff = []
        for p,c in pauli_list:
            self.pPauli.append(p)
            self.pCoeff.append(c)


class ReducedTomography(StandardTomography):
    def _pre_generate_2rdme(self,
            real=True,
            imag=False,
            **kw):
        '''
        focuses more on getting pairs than anything else
        '''
        self.real=real
        self.imag=imag
        self.qubit_pairing = {}
        self._cp = []
        kw['key_list']=self._cp
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []
            # double excitation operators
            ma = {i:j for i,j in enumerate(alp)}
            mb = {i:j for i,j in enumerate(bet)}
            for i in range(2*Na-3):
                a1 = i in alp
                for k in range(i+1,2*Na-2):
                    a2 = k in alp
                    for l in range(k+1,2*Na-1):
                        a3 = l in alp
                        for j in range(l+1,2*Na):
                            a4 = j in alp
                            if not (a1+a2+a3+a4)%2==0:
                                continue
                            idx = '-'.join([str(i),str(k),str(l),str(j)])
                            self.qubit_pairing[idx] = SimplifyTwoBody(
                                    indices=[i,k,l,j],
                                    **kw
                                    )
                            for k1,v in self.qubit_pairing[idx].real.items():
                                self._cp.append(v[0][0])
                                break
            for i in range(2*Na-2):
                for j in range(i+1,2*Na-1):
                    for k in range(j+1,2*Na):
                        idx = '-'.join([str(i),str(j),str(k)])
                        self.qubit_pairing[idx] = SimplifyTwoBody(
                                indices=[i,j,k],
                                **kw
                                )
                        for k1,v in self.qubit_pairing[idx].real.items():
                            self._cp.append(v[0][0])
            # NUMBER NUMBER operator 
            for i in range(2*Na-1):
                for j in range(i+1,2*Na):
                    idx = '-'.join([str(i),str(j)])
                    self.qubit_pairing[idx] = SimplifyTwoBody(
                            indices=[i,j],
                            **kw
                            )
                    for k1,v in self.qubit_pairing[idx].real.items():
                        self._cp.append(v[0][0])
                        break
                        #for p,c in v:
                        #    if p in self._cp:
                        #        pass
                        #    else:
                        #        self._cp.append(p)

    def _generate_2rdme(self,
            real=True,
            imag=False,
            **kw):
        self._pre_generate_2rdme(real=real,imag=imag,Nq=self.Nq,**kw)
        self.real=real
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []
            def sub_rdme(i,k,l,j,spin):
                test = FermionicOperator(
                    coeff=1,
                    indices=[i,k,l,j],
                    sqOp='++--',
                    spin=spin)
                idx = '-'.join([str(i) for i in test.qInd])
                if self.real and not self.imag:
                    try:
                        tomo = self.qubit_pairing[idx].real[test.qOp]
                    except Exception as e:
                        traceback.print_exc()
                        sys.exit()
                elif not self.real and self.imag:
                    try:
                        tomo = self.qubit_pairing[idx].imag[test.qOp]
                    except Exception:
                        pass
                elif self.real and self.imag:
                    tomo = self.qubit_pairing[idx].real[test.qOp]
                    tomoI = self.qubit_pairing[idx].imag[test.qOp]
                    for i in tomoI:
                        tomo.append(i)
                if test.qCo==-1:
                    for i in range(len(tomo)):
                        tomo[i][1] = tomo[i][1]*test.qCo
                return PseudoRDMElement(tomo,test.ind)

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
                            rdme.append(sub_rdme(i,k,l,j,'aaaa'))
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
                            rdme.append(sub_rdme(i,k,l,j,'bbbb'))
            for i in alp:
                for k in bet:
                    for l in bet:
                        for j in alp:
                            if i*Na+k>j*Na+l:
                                continue
                            if imag and i*Na+k==j*Na+l:
                                continue
                            rdme.append(sub_rdme(i,k,l,j,'abba'))
            self.rdme = rdme

    def generate_pauli_measurements(self,simplify=True,**kw):
        paulis = []
        for fermi in self.rdme:
            for j in fermi.pPauli:
                if j in paulis:
                    pass
                else:
                    paulis.append(j)
        if simplify:
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


