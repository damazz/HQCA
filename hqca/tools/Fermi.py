import numpy as np

class FermiOperator:
    def __init__(self,
            coeff=1,
            indices=[0,1,2,3], #orbital indices
            sqOp='-+-+', #second quantized operators
            spin='abba'  # spin, for help.
            ):
        self.c =coeff
        self.ind =indices
        self.op = sqOp
        self.sp = spin
        self.norm = self.c*np.conj(self.c)
        self.order = len(sqOp)
        self.as_set = set(indices)
        self._qubit_order()
        self._classify()

    
    def hasSameInd(self,b):
        '''
        is equal indices
        '''
        if self.qInd==b.qInd:
            return True
        else:
            return False

    def _classify(self):
        if self.order%2==0:
            l = len(set(self.ind))
            if l==1 and self.order==2:
                self.opType = 'no' #number operator
            elif l==2 and self.order==2:
                self.opType = 'se' # single excitation
            elif l==2 and self.order==4:
                self.opType = 'nn' # double number operator
            elif l==3 and self.order==4:
                self.opType = 'ne' # number-excitation operator
            elif l==4 and self.order==4:
                self.opType = 'de' # double excitation operator
            self.rdm = self.order//2
            Nb,Na = 0,0
            for i in self.sp:
                if i=='a':
                    Na+=1 
                elif i=='b':
                    Nb+=1
            if Na-Nb==0:
                self.spBlock = 'ab'
            elif Na-Nb==-1:
                self.spBlock = 'b'
            elif Na-Nb==1:
                self.spBlock = 'a'
            elif Na-Nb==-2:
                self.spBlock = 'bb'
            elif Na-Nb==2:
                self.spBlock = 'aa'
        else:
            pass

    def _qubit_order(self):
        sort = False
        self.no = self.ind[:]
        to = self.op[:]
        ts = self.sp[:]
        ti = self.ind[:]
        tc = self.c
        order = self.order
        while not sort:
            sort=True
            for i in range(order-1):
                if ti[i]>ti[i+1]:
                    tc*=-1
                    to = to[:i]+to[i+1]+to[i]+to[i+2:]
                    ti = ti[:i]+[ti[i+1]]+[ti[i]]+ti[i+2:]
                    ts = ts[:i]+ts[i+1]+ts[i]+ts[i+2:]
                    sort=False
                    break
        self.qOp = to
        self.qInd = ti
        self.qSp = ts
        self.qCo = tc
        self.q2rdm = tc


    def generateExcitationOperators(self,Nq='default',**kw):
        if self.opType=='nn':
            self._NumNumOperator(**kw)
        elif self.opType=='ne':
            self._NumExcOperator(**kw)
        elif self.opType=='de':
            self._ExcExcOperator(**kw)
        elif self.opType=='se':
            self._SingleExcOperator(**kw)

    def _SingleExcOperator(self,Nq='default',**kw):
        self.pauliExp = []
        self.pauliCoeff = []
        if Nq=='default':
            Nq = max(self.qInd)
        c = 1
        if self.qOp=='+-':
            qubSq+= ['XY','YX']
            qubCo+= [c/2,-c/2]
        elif self.qOp=='-+':
            qubSq+= ['XY','YX']
            qubCo+= [c/2,-c/2]
        for item,co in zip(qubSq,qubCo):
            temp = '{}{}{}{}{}{}{}'.format(
                    'I'*self.qInd[0],
                    item[0],
                    'Z'*n1,
                    item[1],
                    'I'*(Nq-1-self.qInd[1]))
            self.pauliExp.append(temp)
            self.pauliCoeff.append(co*self.qCo)

    def _NumExcOperator(self,Nq='default',**kw):
        self.pauliExp = []
        self.pauliCoeff = []
        inds = [
                self.qInd[0]==self.qInd[1],
                self.qInd[1]==self.qInd[2],
                self.qInd[2]==self.qInd[3]
                ]
        if Nq=='default':
            Nq = max(self.qInd)
        qubSq,qubCo=[],[]
        if inds[1]:
            n1= self.qInd[1]-(self.qInd[0]+1)
            n2= self.qInd[3]-(self.qInd[2]+1)
            c=1/4
            qubSq+= ['XIY','XZY','YIX','YZX']
            if self.qOp in ['++--','-+-+']:
                qubCo+= [-c,c,c,-c]
            elif self.qOp in ['+-+-','--++']:
                qubCo+= [c,c,-c,-c]
            for item,co in zip(qubSq,qubCo):
                temp = '{}{}{}{}{}{}{}'.format(
                        'I'*self.qInd[0],
                        item[0],
                        'Z'*n1,
                        item[1],
                        'Z'*n2,
                        item[2],
                        'I'*(Nq-1-self.qInd[3]))
                self.pauliExp.append(temp)
                self.pauliCoeff.append(co*self.qCo)
        elif inds[0]:
            n1= self.qInd[2]-(self.qInd[1]+1)
            n2= self.qInd[3]-(self.qInd[2]+1)
            c = 1/4
            qubSq+= ['IXY','ZXY','IYX','ZYX']
            if self.qOp in ['+-+-','+--+']:
                qubCo+= [+c,-c,-c,+c]
            elif self.qOp in ['-++-','-+-+']:
                qubCo+= [c,c,-c,-c]
            for item,co in zip(qubSq,qubCo):
                temp = '{}{}{}{}{}{}{}'.format(
                        'I'*self.qInd[0],
                        item[0],
                        'Z'*n1,
                        item[1],
                        'Z'*n2,
                        item[2],
                        'I'*(Nq-1-self.qInd[3]))
                self.pauliExp.append(temp)
                self.pauliCoeff.append(co*self.qCo)
        elif inds[2]:
            n1= self.qInd[1]-(self.qInd[0]+1)
            n2= self.qInd[2]-(self.qInd[1]+1)
            c = 1/4
            qubSq+= ['XYI','XYZ','YXI','YXZ']
            if self.qOp in ['+-+-','-++-']:
                qubCo+= [+c,-c,-c,+c]
            elif self.qOp in ['+--+','-+-+']:
                qubCo+= [+c,+c,-c,-c]
            for item,co in zip(qubSq,qubCo):
                temp = '{}{}{}{}{}{}{}'.format(
                        'I'*self.qInd[0],
                        item[0],
                        'Z'*n1,
                        item[1],
                        'Z'*n2,
                        item[2],
                        'I'*(Nq-1-self.qInd[3]))
                self.pauliExp.append(temp)
                self.pauliCoeff.append(co*self.qCo)

    def _ExcExcOperator(self,Nq='default',**kw):
        self.pauliExp = []
        self.pauliCoeff = []
        if Nq=='default':
            Nq = max(self.qInd)
        n1,n3 = self.qInd[1]-(self.qInd[0]+1),self.qInd[3]-(self.qInd[2]+1)
        n2 = self.qInd[2]-(self.qInd[1]+1)
        qubSq,qubCo=[],[]
        c= 1/8
        if self.qOp in ['++--','--++']:
            qubSq+= ['XXXY','XXYX','XYXX','XYYY','YXXX','YXYY','YYXY','YYYX']
            if self.qOp=='--++':
                c*=-1
            qubCo+= [-c,-c,+c,-c,+c,-c,+c,+c]
        elif self.qOp in ['+-+-','-+-+']:
            qubSq+= ['XXXY','XXYX','XYXX','XYYY','YXXX','YXYY','YYXY','YYYX']
            if self.qOp=='-+-+':
                c*=-1
            qubCo+= [+c,-c,+c,+c,-c,-c,+c,-c]
        elif self.qOp in ['+--+','-++-']:
            qubSq+= ['XXXY','XXYX','XYXX','XYYY','YXXX','YXYY','YYXY','YYYX']
            if self.qOp=='-++-':
                c*=-1
            qubCo+= [+c,-c,-c,-c,+c,+c,+c,-c]
        for item,co in zip(qubSq,qubCo):
            temp = '{}{}{}{}{}{}{}{}{}'.format(
                    'I'*self.qInd[0],
                    item[0],
                    'Z'*n1,
                    item[1],
                    'I'*n2,
                    item[2],
                    'Z'*n3,
                    item[3],
                    'I'*(Nq-1-self.qInd[3]))
            self.pauliExp.append(temp)
            self.pauliCoeff.append(co*self.qCo)

    def generateTomoBasis(self,**kw):
        '''
        Note...provides the tomography elements to give the real and/or
        imaginary component of the second-quantized term that is the qubit
        operator. For instance, 
        q+p-, for p<q, will get reorinted into the qubit basis:
            -p-q+ (this is no longer the exact RDM element), and we have:
                self.c = c, self.qCo = -c
        Now, to do tomography on p-q+, we have:
            Re: p-q+ = (p-q+ + (p-q+)^\dag)/2
                     = (p-q+ + q-p+) 
                     = (p-q+ - p+q-)
                     = 1/2 (XX+YY)
                     -> (XX)
        Note, to get our original tomography, we should multiply our result
        times q2rdm, which brings qubit ordered to RDM ordered, which should be
        the original ordering
        '''
        if self.opType=='nn':
            self._NumNumTomography(**kw)
        elif self.opType=='ne':
            self._NumExcTomography(**kw)
        elif self.opType=='de':
            self._ExcExcTomographySimple(**kw)
        
    def _NumNumTomography(self,real=True,imag=False,Nq='default'):
        self.pauliGates = []
        self.pauliCoeff = []
        self.pauliGet = []
        if Nq=='default':
            Nq = max(self.qInd)
        n1= self.qInd[2]-(self.qInd[1]+1)
        qubSq,qubCo=[],[]
        qubGet=[]
        if real:
            qubSq+= ['II','IZ','ZI','ZZ']
            qubGet+=['ZZ','ZZ','ZZ','ZZ']
            if self.qOp in ['+-+-']:
                qubCo+= [1/4,-1/4,-1/4,+1/4]
            elif self.qOp in ['+--+']:
                qubCo+= [1/4,+1/4,-1/4,-1/4]
            elif self.qOp in ['-+-+']:
                qubCo+= [1/4 ,+1/4,+1/4,+1/4]
            elif self.qOp in ['-++-']:
                qubCo+= [1/4,-1/4,+1/4,-1/4]
        for item,co,get in zip(qubSq,qubCo,qubGet):
            temp = '{}{}{}{}{}'.format(
                    'I'*self.qInd[0],
                    item[0],
                    'I'*n1,
                    item[1],
                    'I'*(Nq-1-self.qInd[3]))
            tempGet = '{}{}{}{}{}'.format(
                    'Z'*self.qInd[0],
                    get[0],
                    'Z'*n1,
                    get[1],
                    'Z'*(Nq-1-self.qInd[3]))
            self.pauliGates.append(temp)
            self.pauliCoeff.append(co*self.qCo)
            self.pauliGet.append(tempGet)


    def _NumExcTomography(self,real=True,imag=False,Nq='default'):
        '''
        Important note about ordering, we use the convention:
            Re(p+q+q-r-) = [(A)+(A)^\dag]/2
                         = (p+q+q-r-)+(r+q+q-p-)
                         = (p+q+q-r-)-(p-q+q-r+)
                         -> etc. 

        Note, we did not swap the q+q- operator, due to the unwanted inclusion
        of the 1-RDM element.
        '''
        self.pauliGates = []
        self.pauliCoeff = []
        self.pauliGet = []
        inds = [
                self.qInd[0]==self.qInd[1],
                self.qInd[1]==self.qInd[2],
                self.qInd[2]==self.qInd[3]
                ]
        if Nq=='default':
            Nq = max(self.qInd)
        qubSq,qubCo=[],[]
        qubGet = []
        if inds[1]:
            n1= self.qInd[1]-(self.qInd[0]+1)
            n2= self.qInd[3]-(self.qInd[2]+1)
            if real:
                r = 1/2
                if self.qOp =='++--':
                    qubSq+= ['XIX','XZX']
                    qubGet+=['XZX','XZX']
                    qubCo+= [-r/2,+r/2]
                elif self.qOp =='-+-+':
                    qubSq+= ['XIX','XZX']
                    qubGet+=['XZX','XZX']
                    qubCo+= [+r/2,-r/2]
                elif self.qOp =='+-+-':
                    qubSq+= ['XIX','XZX']
                    qubGet+=['XZX','XZX']
                    qubCo+= [+r/2,+r/2]
                elif self.qOp =='--++':
                    qubSq+= ['XIX','XZX']
                    qubGet+=['XZX','XZX']
                    qubCo+= [-r/2,-r/2]
            if imag:
                c=1j/2
                if self.qOp=='++--':
                    qubGet+=['XZY','XZY']
                    qubSq+= ['XIY','XZY']
                    qubCo+= [-c/2,+c/2]
                elif self.qOp=='-+-+':
                    qubGet+=['XZY','XZY']
                    qubSq+= ['XIY','XZY']
                    qubCo+= [-c/2,+c/2]
                elif self.qOp=='+-+-':
                    qubGet+=['XZY','XZY']
                    qubSq+= ['XIY','XZY']
                    qubCo+= [+c/2,+c/2]
                elif self.qOp=='--++':
                    qubGet+=['XZY','XZY']
                    qubSq+= ['XIY','XZY']
                    qubCo+= [+c/2,+c/2]
            for item,co,get in zip(qubSq,qubCo,qubGet):
                temp = '{}{}{}{}{}{}{}'.format(
                        'I'*self.qInd[0],
                        item[0],
                        'Z'*n1,
                        item[1],
                        'Z'*n2,
                        item[2],
                        'I'*(Nq-1-self.qInd[3]))
                tempGet ='{}{}{}{}{}{}{}'.format(
                        'Z'*self.qInd[0],
                        get[0],
                        'Z'*n1,
                        get[1],
                        'Z'*n2,
                        get[2],
                        'Z'*(Nq-1-self.qInd[3]))
                self.pauliGates.append(temp)
                self.pauliCoeff.append(co*self.qCo)
                self.pauliGet.append(tempGet)
        elif inds[0]:
            n1= self.qInd[2]-(self.qInd[1]+1)
            n2= self.qInd[3]-(self.qInd[2]+1)
            if real:
                r = 1/2
                if self.qOp=='+-+-':
                    qubGet+=['ZXX','ZXX']
                    qubSq+= ['IXX','ZXX']
                    qubCo+= [+r/2,-r/2]
                elif self.qOp=='+--+':
                    qubGet+=['ZXX','ZXX']
                    qubSq+= ['IXX','ZXX']
                    qubCo+= [-r/2,+r/2]
                elif self.qOp=='-++-':
                    qubGet+=['ZXX','ZXX']
                    qubSq+= ['IXX','ZXX']
                    qubCo+= [+r/2,+r/2]
                elif self.qOp=='-+-+':
                    qubGet+=['ZXX','ZXX']
                    qubSq+= ['IXX','ZXX']
                    qubCo+= [-r/2,-r/2]
            if imag:
                c = 1j/2
                if self.qOp=='+-+-':
                    qubGet+=['ZXY','ZXY']
                    qubSq+= ['IXY','ZXY']
                    qubCo+= [c/4,-c/4]
                elif self.qOp=='+--+':
                    qubGet+=['ZXY','ZXY']
                    qubSq+= ['IXY','ZXY']
                    qubCo+= [c/4,-c/4]
                elif self.qOp=='-++-':
                    qubGet+=['ZXY','ZXY']
                    qubSq+= ['IXY','ZXY']
                    qubCo+= [c/4,c/4]
                elif self.qOp=='-+-+':
                    qubGet+=['ZXY','ZXY']
                    qubSq+= ['IXY','ZXY']
                    qubCo+= [c/4,c/4]
            for item,co,get in zip(qubSq,qubCo,qubGet):
                temp = '{}{}{}{}{}{}{}'.format(
                        'I'*self.qInd[0],
                        item[0],
                        'I'*n1,
                        item[1],
                        'Z'*n2,
                        item[2],
                        'I'*(Nq-1-self.qInd[3]))
                tempGet ='{}{}{}{}{}{}{}'.format(
                        'Z'*self.qInd[0],
                        get[0],
                        'Z'*n1,
                        get[1],
                        'Z'*n2,
                        get[2],
                        'Z'*(Nq-1-self.qInd[3]))
                self.pauliGates.append(temp)
                self.pauliCoeff.append(co*self.qCo)
                self.pauliGet.append(tempGet)
        elif inds[2]:
            n1= self.qInd[1]-(self.qInd[0]+1)
            n2= self.qInd[2]-(self.qInd[1]+1)
            if real:
                r = 1/2
                if self.qOp=='+-+-':
                    qubGet+=['XXZ','XXZ']
                    qubSq+= ['XXI','XXZ']
                    qubCo+= [r/2,-r/2]
                elif self.qOp=='-++-':
                    qubGet+=['XXZ','XXZ']
                    qubSq+= ['XXI','XXZ']
                    qubCo+= [-r/2,r/2]
                elif self.qOp=='+--+':
                    qubGet+=['XXZ','XXZ']
                    qubSq+= ['XXI','XXZ']
                    qubCo+= [r/2,r/2]
                elif self.qOp=='-+-+':
                    qubGet+=['XXZ','XXZ']
                    qubSq+= ['XXI','XXZ']
                    qubCo+= [-r/2,-r/2]
            if imag:
                c = 1j/2
                if self.qOp=='+-+-':
                    qubGet+=['XYZ','XYZ']
                    qubSq+= ['XYI','XYZ']
                    qubCo+= [c/2,-c/2]
                elif self.qOp=='-++-':
                    qubGet+=['XYZ','XYZ']
                    qubSq+= ['XYI','XYZ']
                    qubCo+= [c/2,-c/2]
                elif self.qOp=='+--+':
                    qubGet+=['XYZ','XYZ']
                    qubSq+= ['XYI','XYZ']
                    qubCo+= [c/2,c/2]
                elif self.qOp=='-+-+':
                    qubGet+=['XYZ','XYZ']
                    qubSq+= ['XYI','XYZ']
                    qubCo+= [c/2,c/2]
            for item,co,get in zip(qubSq,qubCo,qubGet):
                temp = '{}{}{}{}{}{}{}'.format(
                        'I'*self.qInd[0],
                        item[0],
                        'Z'*n1,
                        item[1],
                        'I'*n2,
                        item[2],
                        'I'*(Nq-1-self.qInd[3]))
                tempGet = '{}{}{}{}{}{}{}'.format(
                        'Z'*self.qInd[0],
                        get[0],
                        'Z'*n1,
                        get[1],
                        'Z'*n2,
                        get[2],
                        'Z'*(Nq-1-self.qInd[3]))
                self.pauliGates.append(temp)
                self.pauliCoeff.append(co*self.qCo)
                self.pauliGet.append(tempGet)

    def _ExcExcTomographySimple(self,real=True,imag=False,Nq='default'):
        self.pauliGates = []
        self.pauliCoeff = []
        self.pauliGet = []
        if Nq=='default':
            Nq = max(self.qInd)
        n1,n3 = self.qInd[1]-(self.qInd[0]+1),self.qInd[3]-(self.qInd[2]+1)
        n2 = self.qInd[2]-(self.qInd[1]+1)
        qubSq,qubCo=[],[]
        if real:
            if self.qOp in ['++--','--++']:
                qubSq+= ['XYXY','XYYX']
                qubCo+= [-1/4,-1/4]
                # -xyxy-xyyx
            elif self.qOp in ['+-+-','-+-+']:
                qubSq+= ['XXYY','XYYX']
                qubCo+= [1/4,1/4]
                # xxyy+xyyx
                pass
            elif self.qOp in ['+--+','-++-']:
                qubSq+= ['XXYY','XYXY']
                qubCo+= [-1/4,-1/4]
                pass
        if imag:
            c= (-1)**(int(self.qOp[0]=='-'))
            if self.qOp in ['++--','--++']:
                qubSq+= ['XYXX','YXXX']
                qubCo+= [c*1j/4,c*1j/4]
                # -xyxy-xyyx
            elif self.qOp in ['+-+-','-+-+']:
                qubSq+= ['XYXX','XXXY']
                qubCo+= [c*1j/4,c*1j/4]
                # xxyy+xyyx
                pass
            elif self.qOp in ['+--+','-++-']:
                qubSq+= ['XXXY','YXXX']
                qubCo+= [c*1j/4,c*1j/4]
        for item,co in zip(qubSq,qubCo):
            temp = '{}{}{}{}{}{}{}{}{}'.format(
                    'I'*self.qInd[0],
                    item[0],
                    'Z'*n1,
                    item[1],
                    'I'*n2,
                    item[2],
                    'Z'*n3,
                    item[3],
                    'I'*(Nq-1-self.qInd[3]))
            tempGet = '{}{}{}{}{}{}{}{}{}'.format(
                    'Z'*self.qInd[0],
                    item[0],
                    'Z'*n1,
                    item[1],
                    'Z'*n2,
                    item[2],
                    'Z'*n3,
                    item[3],
                    'Z'*(Nq-1-self.qInd[3]))
            self.pauliGates.append(temp)
            self.pauliCoeff.append(co*self.qCo)
            self.pauliGet.append(tempGet)

#a = FermiOperator(1,[0,3,3,1],'++--','abba')
#a.generateTomoBasis(real=True,imag=True)
#print(a.pauliGates)
#print(a.pauliCoeff)
#print(a.pauliGet)
