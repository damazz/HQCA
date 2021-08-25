"""
Base quantum string object.
"""

class QuantumString:
    def __init__(self):
        self.s = ''
        self.c = 1.0j

    def norm(self):
        t = ((self.c.real + 1j*self.c.imag)*(self.c.real-1j*self.c.imag))**(0.5)
        return t.real
