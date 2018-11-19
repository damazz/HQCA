import sympy as sy
import numpy as np
import numpy.linalg as LA
import gates as g
from sympy.physics.quantum import TensorProduct as tp
from mpmath import nprint

e1,e2,e3 = sy.symbols('e1,e2,e3')
t1,t2,t3 = sy.symbols('t1,t2,t3')



r1 = sy.Matrix([[sy.cos(t1),-sy.sin(t1)],[sy.sin(t1),sy.cos(t1)]])
r2 = sy.Matrix([[sy.cos(t2),-sy.sin(t2)],[sy.sin(t2),sy.cos(t2)]])
r3 = sy.Matrix([[sy.cos(t3),-sy.sin(t3)],[sy.sin(t3),sy.cos(t3)]])
i4 = sy.eye(4)
i2 = sy.eye(2)
r1_1 = tp(r1,i4)
r1_2 = tp(i2,tp(r1,i2))
r1_3 = tp(i4,r1)
r2_1 = tp(r2,i4)
r2_2 = tp(i2,tp(r2,i2))
r2_3 = tp(i4,r2)
r3_1 = tp(r3,i4)
r3_2 = tp(i2,tp(r3,i2))
r3_3 = tp(i4,r3)

wf = np.matrix([[1],[0],[0],[0],[0],[0],[0],[0]])

wf_1 =np.copy(wf)



def CNOT_13(var,diag=0):
    mat = sy.Matrix([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,diag,var,0,0],
    [0,0,0,0,var,-diag,0,0],
    [0,0,0,0,0,0,diag,var],
    [0,0,0,0,0,0,var,-diag]])
    return mat

def CNOT(var,diag=0):
    mat = sy.Matrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,diag,var],
    [0,0,var,-diag]])
    return mat

def CNOT_12(var,diag=0):
    return np.kron(CNOT(var,diag),np.identity(2))

def CNOT_23(var,diag=0):
    return np.kron(np.identity(2),CNOT(var,diag))

def CNOT_32(var,diag=0):
    return g.g3_H_3*g.g3_H_2*CNOT_23(var,diag)*g.g3_H_3*g.g3_H_2

def CNOT_21(var,diag=0):
    return g.g3_H_2*g.g3_H_1*CNOT_12(var,diag)*g.g3_H_1*g.g3_H_2

def CNOT_31(var,diag=0):
    return g.g3_H_3*g.g3_H_1*CNOT_13(var,diag)*g.g3_H_3*g.g3_H_1


wf_1 = CNOT_21(1) * r1_2 * wf_1
wf_1 = CNOT_32(1) * r2_3 * wf_1
#wf_1 = CNOT_32(1) * r3_3 * wf_1
for a in sy.preorder_traversal(wf_1):
    if isinstance(a,sy.Float):
        wf_1 = wf_1.subs(a,round(a,2))

sy.pprint(wf_1)

'''
ex1 = (2*e1*(1-e1))**(1/2)
ex2 = (2*e2*(1-e2))**(1/2)
ex3 = (2*e3*(1-e3))**(1/2)
ex1 = ex1.subs(e1,0.0276)
ex2 = ex2.subs(e2,0.0266)
ex3 = ex3.subs(e3,0.0233)
#wf_1 = g.g3_X_1*g.g3_X_2*g.g3_X_3*wf_1
wf_1 = (CNOT_13(1-e1,ex1))*r1_1*wf_1
wf_1 = CNOT_12(1-e2,ex2)*r2_1*wf_1
#sy.pprint(wf_1)
wf_1 = CNOT_32(1-e3,ex3)*r3_3*wf_1
wf_1 = wf_1.subs([(e1,0.0276),(e2,0.0266),(e3,0.0233)])
wf_1 = wf_1.subs([(t1,np.pi/4),(t2,np.pi/4),(t3,np.pi/4)])
for a in sy.preorder_traversal(wf_1):
    if isinstance(a,sy.Float):
        wf_1 = wf_1.subs(a,round(a,2))

sy.pprint(wf_1)
#ideal =

wf_2 =np.copy(wf)

#wf_2 = g.g3_X_1*g.g3_X_2*g.g3_X_3*wf_2
wf_2 = (CNOT_13(1,0))*r1_1*wf_2
wf_2 = CNOT_12(1,0)*r2_1*wf_2
wf_2 = CNOT_32(1,0)*r3_3*wf_2
#

wf_2 = wf_2.subs([(t1,np.pi/4),(t2,np.pi/4),(t3,np.pi/4)])
for a in sy.preorder_traversal(wf_2):
    if isinstance(a,sy.Float):
        wf_2 = wf_2.subs(a,round(a,2))

sy.pprint(wf_2)
'''
