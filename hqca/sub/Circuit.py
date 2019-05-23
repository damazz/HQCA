from hqca.sub.BaseRun import QuantumRun
import hqca.quantum.QuantumFunctions as qf
from hqca.quantum.QuantumFramework import build_circuits,run_circuits,Construct
from hqca.tools import Preset as pre

class Quantum(QuantumRun):
    '''
    Class for interfacing with circuits, useful for designing and benchmarking
    '''
    def __init__(self,**kw):
        QuantumRun.__init__(self,**kw)
        self.kw = pre.circuit()
        self.kw_qc = self.kw['qc']
        self.kw_opt = {}

    def build(self):
        QuantumRun._build_quantum(self)
        self._get_ec_pre()
        if self.kw_qc['info'] in ['calc']:
            qf.get_direct_stats(self.QuantStore)
        elif self.kw_qc['info'] in ['draw']:
            qf.get_direct_stats(self.QuantStore,extra='draw')
        elif self.kw_qc['info'] in ['count_only']:
            qf.get_direct_stats(self.QuantStore)
        elif self.kw_qc['info'] in ['compile','check','check_circuit','transpile']:
            qf.get_direct_stats(self.QuantStore,extra='compile')
        else:
            qf.get_direct_stats(self.QuantStore,extra=self.kw_qc['info'])

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
        if self.kw_qc['error_correction'] and self.QuantStore.qc:
            ec_a,ec_b =ec.generate_error_polytope(
                    self.Store,
                    self.QuantStore)
            self.QuantStore.ec_a = ec_a
            self.QuantStore.ec_b = ec_b

    def single(self,para):
        self.QuantStore.para = para
        q_circ,qc_list = build_circuits(QuantStore)
        self.qc_obj = run_circuits(
                q_circ,
                qc_list,
                QuantStore)
        self.proc = Construct(
                self.qc_obj,
                QuantStore)

