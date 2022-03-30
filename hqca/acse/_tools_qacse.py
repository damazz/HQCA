

class Log:
    def __init__(self,
            energy=False,
            norm=False,
            A=False,
            variance=False,
            rdm=False,
            psi=False,
            cnot = False,
            gamma = False,
            opt = False,
            counts= False,
            ):
        self.norm = []
        if energy:
            self.E = []
        if norm:
            self.norm = []
        if variance:
            self.sigma = []
        if rdm:
            self.rdm = []
        if A:
            self.A = []
        if cnot:
            self.cx = []
        if gamma:
            self.gamma = []
        if opt: #log of logs
            self.opts = []
        if counts:
            self.counts = []

