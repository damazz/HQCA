from qiskit import QuantumCircuit
import qiskit
from qiskit import Aer,IBMQ
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from qiskit.transpiler.passes import Unroller



class TranspileQASM:
    def __init__(self,
            qasm
            ):
        try:
            self.qc = QuantumCircuit().from_qasm_file(qasm)
        except FileNotFoundError:
            self.qc = QuantumCircuit().from_qasm_str(qasm)
        print(self.qc)
        self.be = Aer.get_backend('qasm_simulator')

    def get_backend(self,backend):
        IBMQ.load_accounts()
        self.be = IBMQ.get_backend(backend)



    def standard(self,optimization_level=0,limit=None,max_iter=10):
        i=0
        done = False
        if limit==None:
            limit= self.qc.count_ops()['cx']*2
        while not done:
            test = transpile(self.qc,
                    coupling_map=None,
                    backend=self.be,
                    initial_layout=[4,3,0,1,2],
                    optimization_level=optimization_level)
            print(test)
            print(test.count_ops())
            print(test.depth())
            if test.count_ops()['cx']<limit:
                done=True
            if i==max_iter:
                done=True
            i+=1 



