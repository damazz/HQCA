'''
from a given operator, looking to project the operator into the constant
symmetry spaces


'''
from hqca.tools import *
import numpy as np
import sys
from copy import deepcopy as copy
from hqca.tools.quantum_strings import *
import scipy as sp

class NewSymmetryProjection:
    def __init__(self,
            op,
            transform,
            quantstore,
            weight='default',
            skip_sz=False,
            verbose=False):
        pass


class SymmetryProjection:
    def __init__(self,
            op,
            transform,
            quantstore,
            weight='default',
            skip_sz=False,
            verbose=False,
            **kw):
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
        alp,bet = quantstore.alpha['active'],quantstore.beta['active']
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
            ops = []
            first = copy(op[0])
            sites =[]
            for n,i in enumerate(first.s):
                if i in ['+','-']:
                    sites.append(n)
            for j in op:
                E,C = [],[]
                for n,i in enumerate(j.s):
                    if i in ['+','-']:
                        E.append(i)
                    elif i in ['p','h']:
                        C.append(i)
                    ops.append([E,C])
            nEa = 0
            nEb = 0 
            for q in sites:
                if q in alp:
                    nEa+=1
                else:
                    nEb+=1
            if len(ops[0][0])==0:
                self.qubOp = op.transform(transform)
            elif (nEa)%2==1 or (nEb)%2==1:
                # i.e., wrong amount of alpha and beta excitations
                self.qubOp = Operator()+PauliString(0,'I'*op[0].N())
            else:
                perm = Recursive(choices=[ops[0][0]])
                perm.choose()
                perm.simplify()
                dimNe = len(perm.total)   ## length of something? 
                dimCe = 2**(len(ops[0][1])) # length constant
                sz_done = False
                if skip_sz==False:
                    while not sz_done: #filtering out wrong excitations
                        sz_done = True
                        for z,op_str in enumerate(perm.total):
                            na,nb=0,0
                            for s,q in zip(op_str,sites):
                                if q in alp:
                                    na+= (-1)**(s=='+')
                                else:
                                    nb+= (-1)**(s=='+')
                            if not na==0 and not nb==0:
                                perm.total.pop(z)
                                sz_done = False
                                dimNe-=1
                                break
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
                # subroutine using first operator
                # getting Pauli basis 
                # generators 
                #
                temp = copy(first)
                ind = first.inds()
                ts = copy(temp.s)
                op_basis = {}
                dimNull = 0
                initialized=False #propery, 'initialized'
                # ok, so we are making an opeartor basis based 
                # on the permutations generated earlier
                # these are already projected into Sz and N
                for j,p in enumerate(perm.total):
                    for k,q in enumerate(permC):
                        top = ''. join(first.ops()) #temp op
                        m,l=0,0
                        for n in range(len(top)):
                            if top[n] in ['+','-']:
                                top = top[:n]+p[m]+top[n+1:]
                                m+=1 
                            elif top[n] in ['p','h']:
                                top = top[:n]+q[l]+top[n+1:]
                                l+=1 
                        if not initialized:
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
                        if not initialized:
                            initialized=True
                            # if initial, then we generate basis and transformation
                            # matrix pauli_to_op

                            key_list = [o.s for n,o in enumerate(new)]
                            if weight=='default':
                                weight = ['I','X']
                            def count(s):
                                i,x,y,z=0,0,0,0
                                for l in s:
                                    if l=='X':
                                        x+=1
                                    elif l=='Y':
                                        y+=1
                                    elif l=='Z':
                                        z+=1
                                    else:
                                        i+=1
                                return (i,x)
                            key_list = sorted(
                                    key_list,
                                    key=lambda x:count(x),
                                    reverse=True)
                            pauli_basis = {o:n for n,o in enumerate(key_list)}
                            # here, we sort the Pauli basis
                            # sorting here
                            # 
                            pauli_to_op = np.zeros(
                                    (
                                        dimNe*dimCe,
                                        len(pauli_basis.keys())),
                                    dtype=np.complex_)
                        # now, expressing
                        #print(new)
                        for pauli in new:
                            pauli_to_op[
                                    dimCe*j+k-dimNull, #here, null is not added
                                    pauli_basis[pauli.s]
                                    ]=np.conj(pauli.c) #because...it is the reverse transformation
                            #print(pauli)
                            #print(pauli_to_op)
                #print(op_basis)
                #print(pauli_basis)
                #print(pauli_to_op)
                # now, remove null ROWS, not columns
                pauli_to_op = pauli_to_op[:(dimCe*dimNe-dimNull),:]
                #print(pauli_to_op)
                # not pauli to op is a.....something
                # print
                sq_pauli_to_op = np.zeros(
                                    (
                                        dimNe*dimCe-dimNull,
                                        dimNe*dimCe-dimNull),
                                    dtype=np.complex_)
                # optional sorting should be done here
                # 
                # sort sort sort
                #
                #
                # keys = pauli_basis
                #
                #key_list = [k for k,v in pauli_basis

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
                    v_f = np.zeros((dimNe*dimCe-dimNull,1),dtype=np.complex_)
                    #print('----------')
                    for fermi in op:
                        #print(fermi)
                        if fermi.c==0:
                            continue
                        v_f[op_basis[''.join(fermi.ops())]]=fermi.c
                    #print(sq_pauli_to_op)
                    #print(v_f)
                    x = np.linalg.solve(sq_pauli_to_op,v_f)
                    #print(x)
                    n_to_pauli = {v:k for k,v in pauli_basis.items()}
                    #print(n_to_pauli)
                    #print(added)
                    final = Operator()
                    for n,i in enumerate(added):
                        if abs(x[n])>1e-10:
                            if type(x[n]) in [type(np.array([]))]:
                                xn = x[n][0]
                            else:
                                xn = x[n]
                            final+= PauliString(n_to_pauli[i],xn)
                    self.qubOp= final
                    #print(self.qubOp)
                #if first==newop1:
                #    print(self.qubOp)
            #print('-----')
        else:
            sys.exit('Not implemented yet for operators.')
