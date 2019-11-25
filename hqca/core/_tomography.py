from abc import ABC, abstractmethod
from hqca.core._circuit import GenerateCircuit

class Tomography(ABC):
    def __init__(self,
            QuantStore,
            Instructions,
            verbose=True,
            order=2):
        self.qs = QuantStore
        self.order = order
        self.verbose=True
        self.ins = Instructions
        self.circuit_list = []
        self.circuits = []

        pass

    @abstractmethod
    def set(self,**kw):
        '''
        'sets' the problem
        run after generation of problem, typically involves running circuit
        i.e., outlines the tomography and generates circuits to run
        generates rdme elements
        '''
        pass

    @abstractmethod
    def generate(self):
        '''
        generate sthe pauli strings needed for tomgraphy:
            self.op
        '''
        if self.order==2:
            self._generate_2rdme()
        elif self.order==1:
            self._generate_1rdme()
        self._generate_pauli_measurements()

    @abstractmethod
    def construct(self):
        pass

    @abstractmethod
    def simulate(self):
        '''
        Takes:
            self.circuits,
            self.circuit_list

        and runs the objects
        '''
        beo = self.qs.beo
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
                print(job.result().get_unitary(circuit))
                #counts.append(job.result().get_counts(name))
        elif self.qs.backend=='statevector_simulator':
            job = beo.run(qo)
            for circuit in self.circuit_list:
                counts.append(job.result().get_statevector(circuit))
        elif self.qs.use_noise:
            try:
                job = beo.run(
                        qo,
                        backend_options=backend_options,
                        noise_model=noise_model,
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
            except Exception as e:
                print('Error: ')
                print(e)
                traceback.print_exc()
        self.counts = {i:j for i,j in zip(self.circuit_list,counts)}
        

    def _generate_2rdme(self,real=True,imag=False,**kw):
        if not self.grouping:
            alp = self.qs.alpha['active']
            Na = len(alp)
            bet = self.qs.beta['active']
            rdme = []
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
