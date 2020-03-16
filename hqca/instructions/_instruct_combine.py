from hqca.core.primitives import *
import sys
from hqca.core import *

class Combine(Instructions):
    def __init__(self,Ins,*args,**kw):
        self._gates = []
        for ins in Ins:
            temp = ins(*args,**kw)
            for item in temp.gates:
                self._gates.append(item)

    def clear(self):
        self._gates = []

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

