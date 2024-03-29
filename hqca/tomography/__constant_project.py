'''
from a given operator, looking to project the operator into the constant number
space
'''
from hqca.tools import *
import numpy as np
import sys
from copy import deepcopy as copy
from hqca.operators import *
import scipy as sp

class ConstantNumberProjection:
    def __init__(self,op,transform,verbose=False):
        '''
        takes a fermionic operator, and performs the following.

        First, determine the order of E and C within the operators. The set E
        has the elements + and -, and the set C has p and h.

        Then, we look at elements of the operator in the Pauli basis, projected
        onto the constant dimension space, which has dimension of the order of
        E.

        Using this, we can look for a potentially minimal representation if
        there is over redundancy in the basis set. Note we also have to order
        the list of operators that is generated? (Maybe).
        '''
        if isinstance(op,type(QuantumString())):
            E,C=[],[]
            for n,i in enumerate(op.s):
                if i in ['+','-']:
                    E.append(n)
                elif i in ['p','h']:
                    C.append(n)
                nE = len(E)
                nC = len(C)
                op = ''.join(E)
            perm = Recursive(choices=E)
            perm.choose()
            perm.simplify()
            dimNe = len(perm.total)

        elif isinstance(op,type(Operator())):
            first = copy(op[0])
            newop1 = FermiString(coeff=1,
                    indices=[0,4,7,3],
                    ops='++--',
                    N=8)
            newop2 = FermiString(coeff=1,
                    indices=[3,7,4,0],
                    ops='++--',
                    N=8)
            #if first==newop1:
            #    print(first)
            #elif first==newop2:
            #    sys.exit('Found it! 2')
            #print(op)
            ops = []
            for j in op:
                E,C,e = [],[],[]
                for n,i in enumerate(j.s):
                    if i in ['+','-']:
                        E.append(i)
                    elif i in ['p','h']:
                        C.append(i)
                    ops.append([E,C])
            if len(ops[0][0])==0:
                self.qubOp = op.transform(transform)
            else:
                print('here!')
                perm.simplify()
                dimNe = len(perm.total)
                dimCe = 2**(len(ops[0][1]))
                def bin_to_ph(binary):
                    ret = ''
                    for item in binary:
                        if item=='0':
                            ret+='p'
                        else:
                            ret+='h'
                    return ret
                permC = [bin_to_ph(bin(i)[2:]) for i in range(dimCe)]
                new = op.transform(transform)
                if new.null():
                    self.qubOp = new
                    return None
                # 
                # subroutine using first
                # getting Pauli basis 
                # generators 
                #
                temp = copy(first)
                ind = first.inds()
                ts = copy(temp.s)
                op_basis = {}
                dimNull = 0
                initial=False #propery, 'initialized'
                for j,p in enumerate(perm.total):
                    for k,q in enumerate(permC):
                        top = ''. join(first.ops())
                        m,l=0,0
                        for n in range(len(top)):
                            if top[n] in ['+','-']:
                                top = top[:n]+p[m]+top[n+1:]
                                m+=1 
                            elif top[n] in ['p','h']:
                                top = top[:n]+q[l]+top[n+1:]
                                l+=1 
                        if not initial:
                            new = FermiString(
                                    coeff=2**(len(first)),
                                    ops=top,
                                    indices=ind,
                                    N=first.N(),
                                    )
                            new = (Operator()+ new).transform(transform)
                            new = FermiString(
                                    coeff=len(new),
                                    ops=top,
                                    indices=ind,
                                    N=first.N(),
                                    )
                        else:
                            new = FermiString(
                                    coeff=len(pauli_basis.keys()),
                                    ops=top,
                                    indices=ind,
                                    N=first.N(),
                                    )
                        #print(new)
                        new = (Operator()+ new).transform(transform)
                        if dimNe*dimCe>len(new):
                            # i.e., 
                            self.qubOp = op.transform(transform)
                            return None


                        # check for null vectors? 
                        # i.e., because of transformation
                        #print(new)
                        if new.null():
                            dimNull+=1 
                            continue
                        op_basis[top]=j*dimCe+k-dimNull
                        if not initial:
                            initial=True
                            # if initial, then we generate basis and transformation
                            # matrix pauli_to_op
                            pauli_basis = {o.s:n for n,o in enumerate(new)}
                            pauli_to_op = np.zeros(
                                    (
                                        dimNe*dimCe,
                                        len(pauli_basis.keys())),
                                    dtype=np.complex_)
                        # now, expressing
                        for pauli in new:
                            pauli_to_op[
                                    dimCe*j+k-dimNull,
                                    pauli_basis[pauli.s]
                                    ]=pauli.c
                #if first==newop1:
                #    print(dimCe,dimNe,dimNull)
                #    print(pauli_basis)
                #    print(pauli_to_op)

                # now, remove null
                pauli_to_op = pauli_to_op[:(dimCe*dimNe-dimNull),:]
                # not pauli to op is a.....something
                #print(pauli_to_op)
                #print(dimCe,dimNe   )
                # now, look for a square matrix
                sq_pauli_to_op = np.zeros(
                                    (
                                        dimNe*dimCe-dimNull,
                                        dimNe*dimCe-dimNull),
                                    dtype=np.complex_)
                # optional sorting should be done here
                #print(dimNull)
                added = []
                done=False
                #print(pauli_to_op)
                error=False
                while not done:
                    if len(added)==(dimNe*dimCe-dimNull):
                        # then, we are done
                        done=True
                        break
                    elif len(added)==0:
                        sq_pauli_to_op[:,len(added)]=pauli_to_op[:,0]
                        added.append(0)
                        continue
                    for i in range(added[-1],len(pauli_basis.keys())):
                        if i in added:
                            continue
                        vec = pauli_to_op[:,i]
                        use=True
                        # check linear dependence through rank
                        temp = copy(sq_pauli_to_op)
                        temp[:,len(added)]=pauli_to_op[:,i]
                        if not np.linalg.matrix_rank(temp.T)==len(added)+1:
                            use=False
                            continue
                        if use:
                            sq_pauli_to_op[:,len(added)]=vec[:]
                            added.append(i)
                            break
                        else:
                            continue
                    if use:
                        continue
                    else:
                        print('ran into something...?')
                        print(sq_pauli_to_op)
                    # if we reach the end of this iteration, then we are done :
                    print('Could not find linearly independent basis.')
                    done=True
                    error=True
                if error:
                    self.qubOp = op.transform(transform)
                    print('Error generating operator: ')
                    print(op)
                else:
                    # now, express original operator as a vector in op basis
                    v_f = np.zeros((dimNe*dimCe-dimNull,1))
                    for fermi in op:
                        v_f[op_basis[''.join(fermi.ops())]]=fermi.c
                    x = np.linalg.solve(sq_pauli_to_op,v_f)
                    #
                    n_to_pauli = {v:k for k,v in pauli_basis.items()}
                    final = Operator()
                    for n,i in enumerate(added):
                        if abs(x[n])>1e-10:
                            final+= PauliString(n_to_pauli[i],x[n])
                    self.qubOp= final
                #if first==newop1:
                #    print(self.qubOp)
        else:
            sys.exit('Not implemented yet for operators.')






def sign(c):
    if abs(c.real)>1e-14:
        if c.real<0:
            return -1
        elif c.real>0:
            return 1
    elif abs(c.imag)>1e-14:
        if c.imag<0:
            return -1j
        elif c.imag>0:
            return 1j
    else:
        return 0

def trim(b):
    use = []
    for i in range(len(b)):
        if abs(b[i])>1e-10:
            use.append(i)
    return use

class SimplifyTwoBody:
    def __init__(self,
            indices,
            weight='default',
            **kw
            ):
        self.ind = indices
        self.w = weight
        self._sorting_procedure(**kw)
        if len(set(self.ind))==4:
            self._simplify_double_excitation(**kw)
        elif len(set(self.ind))==3:
            self._simplify_number_excitation(**kw)
        elif len(set(self.ind))==2:
            self._simplify_number_operator(**kw)

    def _sorting_procedure(self,
            criteria='pauli-weighting',
            key_list=[],
            **kw,
            ):
        self.crit = criteria
        if self.crit=='pauli-weighting':
            pass
        elif self.crit=='mc':
            # maximally commutative sets
            self.sort_crit = key_list

    def _commutative_paulis(self,A,B):
        k=0
        for i in range(len(A)):
            if not ((A[i]=='I' or B[i]=='I') or A[i]==B[i]):
                k+=1
        return k%2

    def _get_keys(self,x):
        if self.crit=='pauli-weighting':
            if type(self.w)==type('A'):
                return self.fermi_map[x]['weight'][self.w]
            elif type(self.w)==type((1,2)) or type(self.w)==type([1,2]):
                new = []
                for i in self.w:
                    new.append(self.fermi_map[x]['weight'][i])
                return new
        elif self.crit=='mc':
            n=0
            for pauli in self.sort_crit:
                n+=self._commutative_paulis(x,pauli)
            return (n,self.fermi_map[x]['weight']['I'])

    def _simplify_number_operator(self,
            spin='aabb',
            mapping='jw',
            real=True,
            imag=False,
            Nq=0,
            **kw):
        '''
        Given p<q, generate tomography
        '''
        ops = ['ph','hp','pp','hh']
        self.key_ops = {j:i for i,j in zip(range(len(ops)),ops)}
        self.ops = ops
        pauli_map = {}
        for op in ops:
            new = FermionicOperator(
                    coeff=1,
                    indices=self.ind,
                    sqOp=op,
                    spin=spin)
            new.generateOperators(Nq,mapping=mapping,**kw)
            newop = new.formOperator()
            pauli_map[op]={}
            for item in newop.op:
                pauli_map[op][item.p]=sign(item.c)
        fermi_map = {}
        temp = {}
        for so,v in pauli_map.items():
            for p,c in v.items():
                try:
                    temp[p].append(c)
                except Exception:
                    temp[p]=[c]
        for k,v in temp.items():
            v = np.asarray(v)
        for k,v in temp.items():
            fermi_map[k] = {
                    'coeff':v,
                    'real':np.isreal(v[0]),
                    'imag':np.iscomplex(v[0]),
                    'weight':self._weights(k)
                    }
        key_list = [k for k,v in fermi_map.items()]
        self.fermi_map = fermi_map
        if self.w=='default':
            self.w = []
            self.w = ['Z','X']
        self.kl = sorted(
                key_list,
                key=lambda x:self._get_keys(x),
                reverse=True)
        n = len(self.kl)
        lens = len(self.kl[0])
        try:
            done = False
            for l in range(0,n-3):
                if done:
                    continue
                for i in range(l+1,n-2):
                    if done:
                        continue
                    for j in range(i+1,n-1):
                        if done:
                            continue
                        for k in range(j+1,n):
                            v1 = self.fermi_map[self.kl[l]]['coeff']
                            v2 = self.fermi_map[self.kl[i]]['coeff']
                            v3 = self.fermi_map[self.kl[j]]['coeff']
                            v4 = self.fermi_map[self.kl[k]]['coeff']
                            mat = np.matrix([v1,v2,v3,v4])
                            d = np.linalg.det(mat)
                            if abs(d)>1e-10:
                                c1,c2,c3,c4 = copy(l),copy(i),copy(j),copy(k)
                                done = True
                                break
            v1 = self.fermi_map[self.kl[c1]]['coeff']
            v2 = self.fermi_map[self.kl[c2]]['coeff']
            v3 = self.fermi_map[self.kl[c3]]['coeff']
            v4 = self.fermi_map[self.kl[c4]]['coeff']
            mat = np.matrix([v1,v2,v3,v4]).T
            r1 = np.array([1,0,0,0])
            r2 = np.array([0,1,0,0])
            r3 = np.array([0,0,1,0])
            r4 = np.array([0,0,0,1])
            a1 = np.linalg.solve(mat,r1)
            a2 = np.linalg.solve(mat,r2)
            a3 = np.linalg.solve(mat,r3)
            a4 = np.linalg.solve(mat,r4)
            inds = [c1,c2,c3,c4]
            self.real = {
                    'ph':[[self.kl[inds[i]],a1[i]] for i in trim(a1)],
                    'hp':[[self.kl[inds[i]],a2[i]] for i in trim(a2)],
                    'pp':[[self.kl[inds[i]],a3[i]] for i in trim(a3)],
                    'hh':[[self.kl[inds[i]],a4[i]] for i in trim(a4)],
                    }
            self.imag = {}
            for op in ['pp','ph','hp','hh']:
                self.imag[op]=[['I'*lens,0]]
        except Exception as e:
            self.real = {}
            self.imag = {}
            for op in ops:
                new = FermionicOperator(
                        coeff=1,
                        indices=self.ind,
                        sqOp=op,
                        spin=spin)
                new.generateOperators(Nq,mapping=mapping,**kw)
                newop = new.formOperator()
                self.real[op]=[[o.p,o.c] for o in newop.op]
                self.imag[op]=[['I'*lens,0]]


    def _simplify_number_excitation(self,
            spin='aabb',
            real=True,
            imag=False,
            mapping='jw',
            Nq=0,
            **kw):
        self.real = {}
        self.imag = {}
        '''
        Given p<q<r
        '''
        ops = [
                ['+-h','+-p','-+h','-+p',],
                ['+h-','+p-','-h+','-p+',],
                ['h+-','p+-','h-+','p-+',],
                ]
        self.ops =ops
        for place in ops:
            pauli_map = {}
            self.key_ops = {j:i for i,j in zip(range(len(place)),place)}
            for op in place:
                new = FermionicOperator(
                        coeff=1,
                        indices=self.ind,
                        sqOp=op,
                        spin=spin)
                new.generateOperators(Nq,mapping=mapping,**kw)
                newop = new.formOperator()
                pauli_map[op]={}
                # 
                # pauli map is a dict with keys: opeartor, val: paulis 
                # basis of pauli opeartors
                # 
                for item in newop.op:
                    pauli_map[op][item.p]=sign(item.c)
            fermi_map = {}
            temp = {}
            for so,v in pauli_map.items():
                for p,c in v.items():
                    try:
                        temp[p].append(c)
                    except Exception:
                        temp[p]=[c]
            #
            # now, we want to move to the reverse: i.e., key: pauli
            # value: operator
            #
            for k,v in temp.items():
                v = np.asarray(v)
            for k,v in temp.items():
                fermi_map[k] = {
                        'coeff':v,
                        'real':np.isreal(v[0]),
                        'imag':np.iscomplex(v[0]),
                        'weight':self._weights(k)
                        }
            # fermi_map just has dict with:
            # key:val, pauli:sqops
            key_list = [k for k,v in fermi_map.items()]
            self.fermi_map = fermi_map
            # sorting
            if self.w=='default':
                self.w = []
                self.w = ['Z','X']
            self.kl = sorted(
                    key_list,
                    key=lambda x:self._get_keys(x),
                    reverse=True)
            ####### 
            try:
                done=False
                n = len(self.kl)
                for i in range(0,n-1):
                    if done or self.fermi_map[self.kl[i]]['imag']:
                        continue
                    for j in range(i+1,n):
                        if done or self.fermi_map[self.kl[j]]['imag']:
                            continue
                        for k in range(0,n-1):
                            if done or self.fermi_map[self.kl[k]]['real']:
                                continue
                            for l in range(k+1,n):
                                if done or self.fermi_map[self.kl[l]]['real']:
                                    continue
                                v1 = self.fermi_map[self.kl[i]]['coeff']
                                v2 = self.fermi_map[self.kl[j]]['coeff']
                                v3 = self.fermi_map[self.kl[k]]['coeff']
                                v4 = self.fermi_map[self.kl[l]]['coeff']
                                # note, each v is a vector of coefficients in
                                # the op basis 
                                mat = np.matrix([v1,v2,v3,v4])
                                # mat has each row being a different pauli, 
                                # and each col being the op
                                d = np.linalg.det(mat)
                                if abs(d)>1e-10:
                                    c1,c2,c3,c4 = copy(i),copy(j),copy(k),copy(l)
                                    done = True
                                    break
                v1 = self.fermi_map[self.kl[c1]]['coeff']
                v2 = self.fermi_map[self.kl[c2]]['coeff']
                v3 = self.fermi_map[self.kl[c3]]['coeff']
                v4 = self.fermi_map[self.kl[c4]]['coeff']
                mat = np.matrix([v1,v2,v3,v4]).T
                # now, mat is transposed, and so has the row being the op and 
                # cols being the pauli
                # so, r1 is a vec in the op basis
                # and a1, or the solution, is a vec in the pauli basis 
                r1 = np.array([0.5,0,-0.5,0])
                r2 = np.array([0,0.5,0,-0.5])
                i1 = np.array([0.5,0,0.5,0])
                i2 = np.array([0,0.5,0,0.5])
                a1 = np.linalg.solve(mat,r1)
                a2 = np.linalg.solve(mat,r2)
                b1 = np.linalg.solve(mat,i1)
                b2 = np.linalg.solve(mat,i2)
                inds = [c1,c2,c3,c4]  # # c1 is....
                vRe = [a1,a2,-a1,-a2] # this is now...each row is a vector 
                # in the pauli basis, giving the paulis to give that op
                vIm = [-b1,-b2,-b1,-b2] # # 
                for n,op in enumerate(place):
                    # iterating through place with n,
                    # we look to find the real opeartors from vRe that are non
                    # zero, i.e. the correct Pauli matrices
                    self.real[op]=[
                                [self.kl[inds[i]],vRe[n][i]
                                    ] for i in trim(vRe[n])]
                    self.imag[op]=[
                                [self.kl[inds[i]],vIm[n][i]
                                    ] for i in trim(vIm[n])]
            except Exception as e:
                mat = np.asmatrix(
                        [self.fermi_map[v]['coeff'] for v in self.kl]).T
                inds = [0,1,2,3]

                r1 = np.array([0.5,0,-0.5,0])
                r2 = np.array([0,0.5,0,-0.5])
                i1 = np.array([0.5,0,0.5,0])
                i2 = np.array([0,0.5,0,0.5])
                #r1 = np.array([1,0,-1,0])
                #r2 = np.array([0,1,0,-1])
                #i1 = np.array([1,0,1,0])
                #i2 = np.array([0,1,0,1])
                #for vec in [r1,r2,i1,i2]:
                #    try:
                #        x,res,rank,s = np.linalg.lstsq(mat,vec)
                #    except Exception:
                #        sys.exit()
                #        ans = np.linalg.solve(mat,vec)
                #    print(x)
                #    print(res)
                #    print(rank)
                #    print(s)
                a1,res1,rank1,s1 = np.linalg.lstsq(mat,r1)
                a2,res2,rank2,s2 = np.linalg.lstsq(mat,r2)
                b1,res3,rank3,s3 = np.linalg.lstsq(mat,i1)
                b2,res4,rank4,s4 = np.linalg.lstsq(mat,i2)
                vRe = [a1,a2,-a1,-a2]   # #
                vIm = [-b1,-b2,-b1,-b2] # # 
                lens = len(self.kl[0])
                for n,op in enumerate(place):
                    # iterating through place with n,
                    # we look to find the real opeartors from vRe that are non
                    # zero, i.e. the correct Pauli matrices
                    if len(trim(vRe[n]))==0:
                        self.real[op]=[['I'*lens,0]]
                    else:
                        self.real[op]=[
                                    [self.kl[inds[i]],vRe[n][i]
                                        ] for i in trim(vRe[n])]
                    if len(trim(vIm[n]))==0:
                        self.imag[op]=[['I'*lens,0]]
                    else:
                        self.imag[op]=[
                                    [self.kl[inds[i]],vIm[n][i]
                                        ] for i in trim(vIm[n])]
                '''
                conjOp = {
                        '+-h':'-+h',
                        '+-p':'-+p',
                        '-+h':'+-h',
                        '-+p':'+-p',
                        '+h-':'-h+',
                        '+p-':'-p+',
                        '-h+':'+h-',
                        '-p+':'+p-',
                        'h+-':'h-+',
                        'p+-':'p-+',
                        'h-+':'h+-',
                        'p-+':'p+-',}
                '''
                '''
                for op in place:
                    pass
                for op in place:
                    new1 = FermionicOperator(
                            coeff=0.5,
                            indices=self.ind,
                            sqOp=op,
                            spin=spin)
                    new1.generateOperators(Nq,mapping=mapping,**kw)
                    print(new1)
                    new = new1.formOperator()
                    print(new)
                    new2 = FermionicOperator(
                            coeff=-0.5,
                            indices=self.ind,
                            sqOp=conjOp[op],
                            spin=spin)
                    new2.generateOperators(Nq,mapping=mapping,**kw)
                    new += new2.formOperator()
                    new.clean()
                    if len(new.op)==0:
                        self.real[op]=[[Nq*'I',0]]
                    else:
                        self.real[op]=[[o.p,o.c] for o in new.op]
                    # now, imag
                    new1 = FermionicOperator(
                            coeff=0.5,
                            indices=self.ind,
                            sqOp=op,
                            spin=spin)
                    new1.generateOperators(Nq,mapping=mapping,**kw)
                    new = new1.formOperator()
                    new2 = FermionicOperator(
                            coeff=0.5,
                            indices=self.ind,
                            sqOp=conjOp[op],
                            spin=spin)
                    new2.generateOperators(Nq,mapping=mapping,**kw)
                    new += new2.formOperator()
                    new.clean()
                    if len(new.op)==0:
                        self.imag[op]=[[Nq*'I',0]]
                    else:
                        self.imag[op]=[[o.p,o.c] for o in new.op]
                '''

    def _simplify_double_excitation(self,
            spin='aabb',
            real=True,
            imag=False,
            mapping='jw',
            Nq=0,
            **kw):
        '''
        given p<q<r<s
        '''
        ops = [
                '++--','+-+-','+--+',
                '--++','-+-+','-++-']
        self.key_ops = {j:i for i,j in zip(range(len(ops)),ops)}
        self.ops = ops
        pauli_map = {}
        cont = True
        for op in ops:
            # generate the operators 
            new = FermionicOperator(
                    coeff=1,
                    indices=self.ind,
                    sqOp=op,
                    spin=spin,
                    )
            new.generateOperators(Nq,mapping=mapping,**kw)
            newop = new.formOperator()
            pauli_map[op]={}
            if len(newop.op)<=6:
                cont = False
            for item in newop.op:
                pauli_map[op][item.p]=sign(item.c)
        fermi_map = {}
        temp = {}
        for so,v in pauli_map.items():
            #so, second quantized opeartor 
            for p,c in v.items():
                if abs(c)==0:
                    continue
                try:
                    temp[p].append(c)
                except Exception:
                    temp[p]=[c]
        for k,v in temp.items():
            v = np.asarray(v)
        for k,v in temp.items():
            fermi_map[k] = {
                    'coeff':v,
                    'real':np.isreal(v[0]),
                    'imag':np.iscomplex(v[0]),
                    'weight':self._weights(k)
                    }
        key_list = [k for k,v in fermi_map.items()]
        self.fermi_map = fermi_map
        if self.w=='default':
            self.w = []
            self.w = ['Z','X']
        self.kl = sorted(
                key_list,
                key=lambda x:self._get_keys(x),
                reverse=True)
        if len(key_list)==0:
            self.real = {}
            self.imag = {}
            for op in ops:
                lens =  len(list(pauli_map[op].keys())[0])
                self.real[op]=[['I'*lens,0]]
                self.imag[op]=[['I'*lens,0]]
        elif cont:
            self._standard_subproblem()
        else:
            self._degenerate_subproblem()

    def _degenerate_subproblem(self):
        ops = [
                '+-+-','+--+',
                '-+-+','-++-']
        if not len(self.kl)==4:
            sys.exit('Error in degenerate subproblem.')
        done=False
        n = len(self.kl)
        use = [self.kl[0]]
        # get reals
        done = False
        self.f = self.fermi_map
        v1 = self.fermi_map[self.kl[0]]['coeff']
        v2 = self.fermi_map[self.kl[1]]['coeff']
        v3 = self.fermi_map[self.kl[2]]['coeff']
        v4 = self.fermi_map[self.kl[3]]['coeff']
        #print(v1,v2,v3,v4)
        v1 = [i for i in v1 if abs(i)>0]
        v2 = [i for i in v2 if abs(i)>0]
        v3 = [i for i in v3 if abs(i)>0]
        v4 = [i for i in v4 if abs(i)>0]
        mat = np.matrix([v1,v2,v3,v4]).T
        #print(mat)
        r1 = np.array([0.5,0,0.5,0])
        r2 = np.array([0,0.5,0,0.5])
        i1 = np.array([0.5,0,-0.5,0])
        i2 = np.array([0,0.5,0,-0.5])
        a1 = np.linalg.solve(mat,r1)
        a2 = np.linalg.solve(mat,r2)
        b1 = np.linalg.solve(mat,i1)
        b2 = np.linalg.solve(mat,i2)
        inds = [0,1,2,3]  # # c1 is....
        vRe = [a1,a2,a1,a2]   # #
        vIm = [b1,+b2,-b1,-b2] # # 
        self.real = {}
        self.imag = {}
        for n,op in enumerate(ops):
            self.real[op]=[
                        [self.kl[inds[i]],vRe[n][i]
                            ] for i in trim(vRe[n])]
            self.imag[op]=[
                        [self.kl[inds[i]],vIm[n][i]
                            ] for i in trim(vIm[n])]

    def _standard_subproblem(self):
        ops = [
                '++--','+-+-','+--+',
                '--++','-+-+','-++-']
        done=False
        n = len(self.kl)
        use = [self.kl[0]]
        # get reals
        done = False
        self.f = self.fermi_map
        for i in range(0,n-2):
            if self.fermi_map[self.kl[i]]['imag'] or done:
                continue
            for j in range(i+1,n-1):
                if self.fermi_map[self.kl[j]]['imag'] or done:
                    continue
                for k in range(j+1,n):
                    #print(i,j)
                    if self.fermi_map[self.kl[k]]['imag'] or done:
                        continue
                    for p in range(0,n-2):
                        if self.f[self.kl[p]]['real']:
                            continue
                        elif done:
                            continue
                        for q in range(p+1,n-1):
                            if self.f[self.kl[q]]['real']:
                                continue
                            elif done:
                                continue
                            for r in range(q+1,n):
                                if self.f[self.kl[r]]['real']:
                                    continue
                                w1 = self.f[self.kl[i]]['coeff']
                                w2 = self.f[self.kl[j]]['coeff']
                                w3 = self.f[self.kl[k]]['coeff']
                                w4 = self.f[self.kl[p]]['coeff']
                                w5 = self.f[self.kl[q]]['coeff']
                                w6 = self.f[self.kl[r]]['coeff']
                                mat = np.matrix(
                                        [w1,w2,w3,w4,w5,w6],
                                        )
                                d = np.linalg.det(mat)
                                if abs(d)>1e-10:
                                    c1,c2,c3 = copy(i),copy(j),copy(k),
                                    c4,c5,c6 = copy(p),copy(q),copy(r)
                                    done = True
                                    break
        inds = [c1,c2,c3,c4,c5,c6]
        v1 = self.f[self.kl[c1]]['coeff']
        v2 = self.f[self.kl[c2]]['coeff']
        v3 = self.f[self.kl[c3]]['coeff']
        u1 = self.f[self.kl[c4]]['coeff']
        u2 = self.f[self.kl[c5]]['coeff']
        u3 = self.f[self.kl[c6]]['coeff']
        mat = np.matrix([v1,v2,v3,u1,u2,u3]).T
        r1 = np.array([0.5,0,0,+0.5,0,0])
        r2 = np.array([0,0.5,0,0,+0.5,0])
        r3 = np.array([0,0,0.5,0,0,+0.5])
        i1 = np.array([0.5,0,0,-0.5,0,0])
        i2 = np.array([0,0.5,0,0,-0.5,0])
        i3 = np.array([0,0,0.5,0,0,-0.5])
        a1 = np.linalg.solve(mat,r1)
        a2 = np.linalg.solve(mat,r2)
        a3 = np.linalg.solve(mat,r3)
        b1 = np.linalg.solve(mat,i1)
        b2 = np.linalg.solve(mat,i2)
        b3 = np.linalg.solve(mat,i3)
        #ops = [
        #        '++--','+-+-','+--+',
        #        '--++','-+-+','-++-']
        vRe = [a1,a2,a3,a1,a2,a3]   # #
        vIm = [-b1,-b2,-b3,b1,b2,b3] # # 
        self.real = {}
        self.imag = {}
        for n,op in enumerate(ops):
            self.real[op]=[
                        [self.kl[inds[i]],vRe[n][i]
                            ] for i in trim(vRe[n])]
            self.imag[op]=[
                        [self.kl[inds[i]],vIm[n][i]
                            ] for i in trim(vIm[n])]
        #self.real = {
        #        '++--':[[self.kl[inds[i]],a1[i]] for i in trim(a1)],
        #        '--++':[[self.kl[inds[i]],a1[i]] for i in trim(a1)],
        #        '+-+-':[[self.kl[inds[i]],a2[i]] for i in trim(a2)],
        #        '-+-+':[[self.kl[inds[i]],a2[i]] for i in trim(a2)],
        #        '+--+':[[self.kl[inds[i]],a3[i]] for i in trim(a3)],
        #        '-++-':[[self.kl[inds[i]],a3[i]] for i in trim(a3)],
        #        }
        #self.imag = {
        #        '++--':[[self.kl[inds[i]],+1*b1[i]] for i in trim(b1)],
        #        '--++':[[self.kl[inds[i]],-1*b1[i]] for i in trim(b1)],
        #        '+-+-':[[self.kl[inds[i]],+1*b2[i]] for i in trim(b2)],
        #        '-+-+':[[self.kl[inds[i]],-1*b2[i]] for i in trim(b2)],
        #        '+--+':[[self.kl[inds[i]],+1*b3[i]] for i in trim(b3)],
        #        '-++-':[[self.kl[inds[i]],-1*b3[i]] for i in trim(b3)],
        #        }



    def _weights(self,string):
        count = {'I':0,'X':0,'Y':0,'Z':0}
        for i in string:
            count[i]+=1
        return count



