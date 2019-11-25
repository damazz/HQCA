

class QubitOperator:
    def __init__(self,
            pauli,
            coeff):
        self.p = pauli
        self.c = coeff
        self.g = pauli #get 

    def __eq__(self,a):
        return self.p==a.p

    def __neq__(self,a):
        return self.p==a.p

    def clear(self):
        pass

