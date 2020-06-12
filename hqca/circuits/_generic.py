from hqca.core import *

class GenericCircuit(Circuit):
    def __init__(self,**kwargs):
        Circuit.__init__(self,**kwargs)
        pass

    def apply(self,**kwargs):
        Circuit.apply(self,**kwargs)

    def h(self,q):
        self.qc.h(self.q[q])

    def s(self,q):
        self.qc.s(self.q[q])

    def si(self,q):
        self.qc.sdg(self.q[q])

    def Cx(self,q,p):
        self.qc.cx(self.q[q],self.q[p])

    def Cz(self,q,p):
        self.qc.cz(self.q[q],self.q[p])

    def Rx(self,q,val):
        self.qc.rx(val,self.q[q])

    def Ry(self,q,val):
        self.qc.ry(val,self.q[q])

    def Rz(self,q,val):
        self.qc.rz(val,self.q[q])
