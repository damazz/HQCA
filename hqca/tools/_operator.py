from copy import deepcopy as copy

class Operator:
    def __init__(self,
            ops = [],
            antihermitian=True):
        self._op = ops
        self.ah=True

    def _update(self):
        pass

    def __add__(self,A):
        '''
        Commutative addition
        '''
        temp = copy(self)
        for old_op in temp._op:
            old_op.clear()
        try:
            for new in A._op:
                new_op =True
                for old in temp._op:
                    if old==new:
                        try:
                            old.qCo = new.qCo+old.qCo
                        except Exception:
                            pass
                        old.c  = new.c+old.c
                        new_op = False
                        break
                    elif old!=new:
                        if self.ah:
                            try:
                                old.qCo = -new.qCo+old.qCo
                            except Exception:
                                pass
                            old.c  = -new.c+old.c
                        else:
                            try:
                                old.qCo = new.qCo+old.qCo
                            except Exception:
                                pass
                            old.c  = new.c+old.c
                        new_op = False
                        break
                if new_op:
                    temp._op.append(new)
        except AttributeError:
            new_op =True
            for old in self._op:
                if old==A:
                    try:
                        old.qCo = A.qCo+old.qCo
                    except Exception:
                        pass
                    old.c  = A.c+old.c
                    new_op = False
                    break
                elif old_op!=A:
                    if self.ah:
                        try:
                            old.qCo = -A.qCo+old.qCo
                        except Exception:
                            pass
                        old.c  = -A.c+old.c
                    else:
                        try:
                            old.qCo = A.qCo+old.qCo
                        except Exception:
                            pass
                        old.c  = A.c+old.c
                    new_op = False
                    break
            if new_op:
                temp._op.append(A)
        return temp
    
    def __iadd__(self,A):
        temp = self+A
        self._op = temp._op
        return self

    def generateSkewExpOp(self):
        for items in self._op:
            items.generateAntiHermitianExcitationOperators()

    def generateHermExpOp(self):
        for items in self._op:
            items.generateHermitianExcitationOperators()

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self,nop):
        self._op = nop

