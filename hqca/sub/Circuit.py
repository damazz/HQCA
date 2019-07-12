from hqca.sub.BaseRun import QuantumRun
import hqca.quantum.QuantumFunctions as qf
from hqca.quantum.QuantumFramework import build_circuits,run_circuits,Construct
from hqca.tools import Preset as pre
import numpy as np
from hqca.quantum import NoiseSimulator as ns

class Quantum(QuantumRun):
    '''
    Class for interfacing with circuits, useful for designing and benchmarking
    '''
    def __init__(self,theory='noft',**kw):
        QuantumRun.__init__(self,**kw)
        self.theory=theory
        self.kw = pre.circuit()
        self.kw_qc = self.kw['qc']
        self.kw_opt = {}

    def build(self):
        QuantumRun._build_quantum(self)
        qf.get_direct_stats(self.QuantStore)
        if self.QuantStore.ec_pre:
            if self.QuantStore.filter_meas:
                ns.get_measurement_filter(self.QuantStore)

    def _get_ec_pre(self):
        '''
        Method for getting error correction pre-runs

        Includes:
            - extrapolative procedures
            - triangulation procedure
        '''
        try:
            if self.kw_qc['tri']:
                self.kw_qc['triangle']=tri.find_triangle(
                        Ntri=self.kw_qc['method_Ntri'],
                        **self.kw_qc)
        except KeyError:
            pass

    def single(self,para):
        self.QuantStore.parameters = para
        q_circ,qc_list = build_circuits(self.QuantStore)
        self.qc_obj = run_circuits(
                q_circ,
                qc_list,
                self.QuantStore)
        self.proc = Construct(
                self.qc_obj,
                self.QuantStore)
        self.proc.find_signs()
        Nso = self.proc.rdm1.shape[0]
        rdma = self.proc.rdm1[0:Nso//2,0:Nso//2]
        rdmb = self.proc.rdm1[Nso//2:,Nso//2:]
        self.noca,self.nora = np.linalg.eig(rdma)
        self.nocb,self.norb = np.linalg.eig(rdmb)


