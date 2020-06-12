from hqca.core.primitives import *
import sys
from hqca.core import *
import numpy as np
from hqca.tools import *

class SingleQubitExponential(Instructions):
    '''
    Complex instructions: can implement arbitrary 1- body unitary
    transformations as exact operators. There are also additional components for
    compilation of different gates. Indeed, for a generic op, we compile a
    string into successive rotations. The simple option will compile each step
    into one rotation.

    Propagation should be specified as well, as either:
        1) trotter
        2) rotation
        3) commute

    'trotter' applies the default gate sequence, as separate rotations. 
    'rotation' applies a single uniary rotation, i.e. an exact form of the the
    exponentiated Hamiltonian
    'commute' applies the rotated single unitary Hamiltonian, but then also
    adds it into the ansatz, resulting in only one gate application. 
    ''' 
    def __init__(self,
            operator,
            Nq,
            propagate=False,
            initial_state=[],
            simple=False,
            **kw
            ):
        self._gates = []
        self._applyOp(operator,Nq,simple,**kw)
        if propagate:
            self._applyH(**kw)

    def clear(self):
        self._gates = []

    def _applyOp(self,operator,Nq,simple=False,**kw):
        if simple:
            self._applySimpleOp(operator,Nq)
        else:
            self._applyGenericOp(operator,Nq)

    def _applyGenericOp(self,operator,Nq,
            **kw):
        #print('Operator')
        #print(operator)
        if not Nq==1:
            sys.exit('Not correct Instructions!')
        new = Operator()
        x,y,z=0,0,0
        for item in operator:
            if abs(item.c)<1e-10:
                continue
            if item.s=='Z':
                z+=1 
            elif item.s=='X':
                x+=1 
            elif item.s=='Y':
                y+=1 
            if z>1 or x>1 or y>1:
                self._xyz_to_rotation(new)
                new = Operator()
                x,y,z = 0,0,0
                if item.p=='Z':
                    z+=1
                elif item.p=='X':
                    x+=1
                elif item.p=='Y':
                    y+=1
            new+=item
        self._xyz_to_rotation(new)

    def _applySimpleOp(self,operator,Nq,**kw):
        #print('Operator')
        #print(operator)
        if not Nq==1:
            sys.exit('Not correct Instructions!')
        rotations = []
        new = Operator()
        x,y,z=0,0,0
        for item in operator:
            if abs(item.c)<1e-10:
                continue
            if item.s=='Z':
                z+=1 
            elif item.s=='X':
                x+=1 
            elif item.s=='Y':
                y+=1 
            if z>1 or x>1 or y>1:
                rotations.append(new)
                new = Operator()
                x,y,z = 0,0,0
                if item.p=='Z':
                    z+=1
                elif item.p=='X':
                    x+=1
                elif item.p=='Y':
                    y+=1
            new+=item
        rotations.append(new)
        while not len(rotations)==1:
            op2 = rotations.pop(-1)
            op1 = rotations.pop(-1)
            rotations.append(
                    self._combine_xyz_abc_rotation(op1,op2)
                    )
        self.op = rotations[0]
        self._xyz_to_rotation(self.op)

    def _combine_xyz_abc_rotation(self,op1,op2,s1=1,s2=1):
        '''
        following stadard law of spherical cosines, see Wikipedia page on the
        commutation relations of Pauli matrices
        '''
        v1 = {'I':0,'X':0,'Y':0,'Z':0}
        v2 = {'I':0,'X':0,'Y':0,'Z':0}
        #print('Combining operators')
        for op in op1:
            if abs(op.c.imag)<1e-12:
                v1[o.s]+=op.c.real
            else:
                v1[op.s]+=op.c.imag
        for op in op2:
            if abs(op.c.imag)<1e-12:
                v2[op.s]+=op.c.real
            else:
                v2[op.s]+=op.c.imag
        n = np.matrix([
            [v1['X']],
            [v1['Y']],
            [v1['Z']]])
        m = np.matrix([
            [v2['X']],
            [v2['Y']],
            [v2['Z']]])
        a= np.linalg.norm(n)
        b = np.linalg.norm(m)
        if abs(a)<1e-14:
            new = Operator()
            new+= PauliString('X',m[0,0]*s2)
            new+= PauliString('Y',m[1,0]*s2)
            new+= PauliString('Z',m[2,0]*s2)
        elif abs(b)<1e-14:
            new = Operator()
            new+= PauliString('X',n[0,0]*s1)
            new+= PauliString('Y',n[1,0]*s1)
            new+= PauliString('Z',n[2,0]*s1)
        else:
            n*=(1/np.linalg.norm(a))
            m*=(1/np.linalg.norm(b))
            a*=s1
            b*=s2
            dot = np.dot(np.conj(n.T),m)[0,0]
            c = np.arccos(
                    np.cos(a)*np.cos(b)-dot*np.sin(a)*np.sin(b))
            term = n*np.sin(a)*np.cos(b)
            term+= m*np.sin(b)*np.cos(a)
            term-= np.cross(n,m,axis=0)*np.sin(a)*np.sin(b)
            k = (1/np.sin(c))*term*c
            #print('New k')
            new = Operator()
            new+= PauliString('X',k[0,0])
            new+= PauliString('Y',k[1,0])
            new+= PauliString('Z',k[2,0])
        return new

    def _xyz_to_rotation(self,operator,scale=1):
        val = {'I':0,'X':0,'Y':0,'Z':0}
        for op in operator:
            if abs(op.c.imag)<1e-12:
                val[op.s]+=op.c.real
            else:
                val[op.s]+=op.c.imag
        #####
        norm = np.conj(val['X'])*val['X']
        norm+= np.conj(val['Y'])*val['Y']
        norm+= np.conj(val['Z'])*val['Z']
        if abs(norm)<1e-14:
            return
        norm = np.sqrt(norm)
        theta=  2*norm*scale
        c,s = np.cos(theta/2),np.sin(theta/2)
        new = (-1j*s/norm)*np.matrix([
            [val['Z'],(val['X']-1j*val['Y'])],
            [(val['X']+1j*val['Y']),-val['Z']]
            ])
        #print('Pauli rot matrix')
        ##print(new)
        new+= np.identity(2)*c
        #print('Norm: {}, Theta: {}'.format(norm,theta))
        #print(val)
        ##print('Rotation matrix: ')
        #print(new)
        gam = 2*np.arccos(max(-1,min(np.abs(new[0,0]),1)))
        cg,sg = np.cos(gam/2),np.sin(gam/2)
        da = 1j*(new[1,1]-new[0,0])
        cb = 1*(new[1,0]-new[0,1])
        #print('d minus a: {}'.format(da))
        #print('c minus b: {}'.format(cb))
        m = np.arcsin(max(-1,min(1,-da/(2*cg))))
        n = np.arccos(max(-1,min(1,cb/(2*sg))))
        #print('m: {}'.format(m))
        #print('n: {}'.format(n))
        beta = m+n
        delt = m-n
        test = Circ(1)
        test.Rz(0,beta)
        test.Ry(0,gam)
        test.Rz(0,delt)
        # if test 
        if np.linalg.norm(test.m-new)>1e-10:
            gam*=-1
            cg,sg = np.cos(gam/2),np.sin(gam/2)
            da = 1j*(new[1,1]-new[0,0])
            cb = 1*(new[1,0]-new[0,1])
            #print('d minus a: {}'.format(da))
            #print('c minus b: {}'.format(cb))
            m = np.arcsin(max(-1,min(1,-da/(2*cg))))
            n = np.arccos(max(-1,min(1,cb/(2*sg))))
            #print('m: {}'.format(m))
            #print('n: {}'.format(n))
            beta = m+n
            delt = m-n

            test = Circ(1)
            test.Rz(0,beta)
            test.Ry(0,gam)
            test.Rz(0,delt)
        if np.linalg.norm(test.m-new)>1e-6:
            print('Error in matrix.')
            print(np.linalg.norm(test.m-new))
            print(test.m)
            print(new)
            sys.exit('')
        if abs(val['I'])>1e-10:
            self._gates.append(
                    [(val['I'],'I'),generic_Pauli_term])
        self._gates.append(
                [(beta,'Z'),generic_Pauli_term])
        self._gates.append(
                [(gam,'Y'),generic_Pauli_term])
        self._gates.append(
                [(delt,'Z'),generic_Pauli_term])

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

    def _applyH(self,
            propagate_method='trotter',
            **kw):
        if propagate_method=='trotter':
            self._applyH_trotter(**kw)
        elif propagate_method=='rotation':
            self._applyH_rotate(**kw)
        elif propagate_method=='commute':
            self._applyH_simplify(**kw)
        else:
            print('Error in _applyH')
            print('Incorrect propagation specified, use:')
            print('1) trotter, 2) rotation, or 3) commute')
            sys.exit('Goodbye.')

    def _applyH_simplify(self,
            HamiltonianOperator,
            trotter_steps=1,
            scaleH=0.5,**kw):
        if trotter_steps>1:
            sys.exit('No need for trotter steps.')
        try:
            self.op
        except Exception:
            sys.exit('Did not simplify circuit before applying Hamiltonian.')
        self.clear()
        self._xyz_to_rotation(
                self._combine_xyz_abc_rotation(
                    self.op,
                    HamiltonianOperator,
                    s1=1,
                    s2=scaleH
                    )
                )

    def _applyH_rotate(self,
            HamiltonianOperator,
            trotter_steps=1,
            scaleH=0.5,**kw):
        if trotter_steps>1:
            sys.exit('No need for trotter steps.')
        else:
            print('Applying Hamiltonian.')
            self._xyz_to_rotation(HamiltonianOperator,scale=scaleH)

    def _applyH_trotter(self,
            HamiltonianOperator,
            trotter_steps=1,
            scaleH=0.5,**kw):
        for i in range(trotter_steps):
            for item in HamiltonianOperator.op:
                self._gates.append(
                        [(
                            (1/trotter_steps)*scaleH*item.c,
                            item.s
                            ),
                            generic_Pauli_term
                            ]
                        )

class TwoQubitExponential(Instructions):
    def __init__(self):
        pass
