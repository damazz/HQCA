"""
Abstract Tomography class. Used as a basis forRDMs in StandardTomography. Has the properties:
set
generate
simulate
construct
"""
from abc import ABC, abstractmethod


class Tomography(ABC):
    def __init__(self,
            quantstore,
            instruct=None,
            verbose=True,
            order=2):
        self.qs = quantstore
        self.order = order
        self.verbose=True
        self.ins = instruct
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
        pass


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
        pass
