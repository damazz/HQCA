from hqca.core import *

class GenericCircuit(Circuit):
    def __init__(self,**kwargs):
        Circuit.__init__(self,**kwargs)
        pass

    def apply(self,**kwargs):
        Circuit.apply(self,**kwargs)
