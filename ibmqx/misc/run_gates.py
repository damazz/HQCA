import sys
import gates as g
import numpy as np
import sympy as sy

np.set_printoptions(precision=3)
t1,t2 = sy.symbols('t1,t2')
al,be,ga,de = sy.symbols('al,be,ga,de')
a,b,c,d,e,f,h,i = sy.symbols('a,b,c,d,e,f,h,i')
test = sy.Matrix([[al,0],[0,de]])
test = np.kron(np.identity(4),test)
vec = sy.Matrix([[a,b,c,d,e,f,h,i]]).T
R_y1 = sy.Matrix([[sy.cos(t1),-sy.sin(t1)],[sy.sin(t1),sy.cos(t1)]])
R_y2 = sy.Matrix([[sy.cos(t2),-sy.sin(t2)],[sy.sin(t2),sy.cos(t2)]])
M1 = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
M2 = np.matrix([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
R_y1_1 = np.kron(R_y1,np.identity(2))
R_y2_1 = np.kron(R_y2,np.identity(2))
R_y1_2 = np.kron(np.identity(2),R_y1)
R_y2_2 = np.kron(np.identity(2),R_y2)
rC = g.g2_H_1*g.g2_H_2*g.CNOT*g.g2_H_1*g.g2_H_2
sq = 1/np.sqrt(2)


def get(mat):
    a = ''
    for i in range(0,len(mat)):
        if mat[i,i]==-1:
            a+= 'c{} '.format(i)
    print(a)

#print(g.g2_C10*g.g2_Cz)


#get(g.g3_C00_23*g.g3_Cz_13*-g.g3_Z_1*g.g3_Z_2)
a = g.g3_Cz_23*g.g3_C10_23
a = g.g3_Z_2
#print(g.g3_C01_12*g.g3_Cz_12)
#print(g.g3_Z_2)
#a = g.g3_Cz_12*g.g3_Cz_13*g.g3_C00_23
#a = g.g3_Cz_13*g.g3_Cz_12*g.g3_C10_23*g.g3_C00_12*g.g3_C00_23
#get(a)
#a = g.g3_Cz_12*g.g3_Cz_23*g.g3_Cz_13
#b = g.g4_Cz_12*g.g4_Cz_13*g.g4_Cz_14*g.g4_Cz_24*g.g4_Cz_34*g.g4_Cz_23
#c = g.g4_Cz_23*g.g4_Cz_13
#d = g.g4_Cz_34*g.g4_Cz_24*g.g4_Cz_14*g.g4_Cz_13*g.g4_Cz_23*g.g4_Cz_12


#print(a)
#print(g.g3_Cz_12)

sys.exit()
################
vec = g.g3_Cz_12*vec
Rh = np.matrix([[sq,-sq],[sq,sq]])
Rh_2 = np.kron(np.identity(2),Rh)
Rh_1 = np.kron(Rh,np.identity(2))
D16 = vec.T *g.rdm16 * vec
D25 = vec.T *g.rdm25 * vec
D34 = vec.T *g.rdm34 * vec
print(D16)
print(D25)
print(D34)

sys.exit()
#################
alp = g.g2_Cz*g.g2_Z_2*g.g2_H_2*g.g2_Cz
bet1 = g.g2_H_1*g.g2_Z_1
bet2 = g.g2_H_2*g.g2_Z_2
bet3 = g.g2_Z_2*g.g2_H_2
bet2_mod = g.g2_Cz*g.g2_Z_2*g.g2_H_2*g.g2_Cz
gam1 = g.g2_H_1*g.g2_Z_1
gam2 = g.g2_H_2*g.g2_Z_2
alpm = g.g2_X_1*g.g2_X_2*g.g2_Z_1
al2 = g.g2_Z_2*g.g2_H_2
test = rC*g.g2_H_2*g.g2_Z_2*rC
#test = gam1*gam2
#test = np.identity(4)
#print(g.g2_H_1*g.g2_H_2*g.CNOT*g.g2_H_1*g.g2_H_2*g.g2_H_2*g.g2_Z_2*g.g2_H_1*g.g2_H_2*g.CNOT*g.g2_H_1*g.g2_H_2)
#print(g.CNOT*g.g2_Z_1)
#print(gam1)
#print(g.g2_Z_1)
#print(g.g2_X_2*M2*g.g2_X_2)
#print(vec.T*test.T*M2*test*vec)

#print(alpm)

#print(A)
#print(R_y1_1)

#print((A-B))


