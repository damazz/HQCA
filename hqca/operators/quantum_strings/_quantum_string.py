"""
Base quantum string object.
"""

class QuantumString:
    def __init__(self):
        self.s = ''
        self.c = 1.0j

    def norm(self):
        return (self.c*(self.c.real-self.c.imag))**(0.5)
