
from hqca.core import *
from hqca.tools import *
from hqca.vqe._store_vqe import *
from hqca.tomography import *
from hqca.vqe._ucc import *
from hqca.vqe._pair import *
from copy import deepcopy as copy


class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False

class RunADAPTVQE(QuantumRun):
    def __init__(self,
            Storage,
            Optimizer,
            QuantStore,
            Instructions,
            **kw,
            ):
        self.Store = Storage
        self.Opt = Optimizer
        self.QuantStore = QuantStore
        self.Instruct = Instructions
        self._update_vqe_kw(**kw)
