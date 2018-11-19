import numpy as np

CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
rCNOT = np.matrix([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
r = 1/(np.sqrt(2))
g1_H = np.matrix([[r,r],[r,-r]])
g2_H_1 = np.kron(g1_H,np.identity(2))
g2_H_2 = np.kron(np.identity(2),g1_H)
g3_H_1 = np.kron(g1_H,np.identity(4))
g3_H_2 = np.kron(np.identity(2),g2_H_1)
g3_H_3 = np.kron(np.identity(4),g1_H)

g2_Cz = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
g3_Cz_12 = np.kron(g2_Cz,np.identity(2))
g3_Cz_23 = np.kron(np.identity(2),g2_Cz)
g3_Cz_13 = np.matrix([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,-1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,-1]])


g4_Cz_12 = np.kron(g2_Cz,np.identity(4))
g4_Cz_23 = np.kron(g3_Cz_23,np.identity(2))
g4_Cz_13 = np.kron(g3_Cz_13,np.identity(2))
g4_Cz_14 = np.identity(16)
g4_Cz_14[9,9]=-1
g4_Cz_14[11,11]=-1
g4_Cz_14[13,13]=-1
g4_Cz_14[15,15]=-1

g4_Cz_24 = np.kron(np.identity(2),g3_Cz_13)
g4_Cz_34 = np.kron(np.identity(4),g2_Cz)


g3_CNOT_12 = np.kron(CNOT,np.identity(2))
g3_CNOT_21 = g3_H_1*g3_H_2*g3_CNOT_12*g3_H_1*g3_H_2

g3_CNOT_23 = np.kron(np.identity(2),CNOT)
g3_CNOT_32 = g3_H_3*g3_H_2*g3_CNOT_23*g3_H_3*g3_H_2


g3_CNOT_13 = np.matrix([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0]])

g3_CNOT_31= g3_H_3*g3_H_1*g3_CNOT_13*g3_H_3*g3_H_1

rdm16 = np.matrix([
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0]])

rdm25 = np.matrix([
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [-1,0,0,0,0,0,0,0],
    [0,-1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,-1,0,0,0],
    [0,0,0,0,0,-1,0,0]])

rdm34 = np.matrix([
    [0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0]])

g2_S = np.matrix([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]])
g3_S_12 = np.kron(g2_S,np.identity(2))
g3_S_23 = np.kron(np.identity(2),g2_S)
g3_S_13 = np.matrix([
    [1,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,1]])

g1_Y = np.matrix([[0,-1j],[1j,0]])
g1_X = np.matrix([[0,1],[1,0]])
g1_Z = np.matrix([[1,0],[0,-1]])

g2_Z_1 = np.kron(g1_Z,np.identity(2))
g2_Y_1 = np.kron(g1_Y,np.identity(2))
g2_X_1 = np.kron(g1_X,np.identity(2))

g2_Z_2 = np.kron(np.identity(2),g1_Z)
g2_Y_2 = np.kron(np.identity(2),g1_Y)
g2_X_2 = np.kron(np.identity(2),g1_X)

g3_Z_1 = np.kron(g1_Z,np.identity(4))
g3_Y_1 = np.kron(g1_Y,np.identity(4))
g3_X_1 = np.kron(g1_X,np.identity(4))

g3_Z_3 = np.kron(np.identity(4),g1_Z)
g3_Y_3 = np.kron(np.identity(4),g1_Y)
g3_X_3 = np.kron(np.identity(4),g1_X)


g3_Z_2 = np.kron(np.identity(2),np.kron(g1_Z,np.identity(2)))
g3_Y_2 = np.kron(np.identity(2),np.kron(g1_Y,np.identity(2)))
g3_X_2 = np.kron(np.identity(2),np.kron(g1_X,np.identity(2)))

g2_C00 = g2_X_1*g2_X_2*g2_Cz*g2_X_1*g2_X_2

g3_C00_12 = np.kron(g2_C00,np.identity(2))
g3_C00_23 = np.kron(np.identity(2),g2_C00)
g3_C00_13 = np.identity(8)
g3_C00_13[0,0]=-1
g3_C00_13[2,2]=-1
#
g2_C01 = g2_X_1*g2_Cz*g2_X_1

g3_C01_12 = np.kron(g2_C01,np.identity(2))
g3_C01_23 = np.kron(np.identity(2),g2_C01)
g3_C01_13 = np.identity(8)
g3_C01_13[1,1]=-1
g3_C01_13[3,3]=-1

#
g2_C10 = g2_X_2*g2_Cz*g2_X_2

g3_C10_12 = np.kron(g2_C10,np.identity(2))
g3_C10_23 = np.kron(np.identity(2),g2_C10)
g3_C10_13 = np.identity(8)
g3_C10_13[4,4]=-1
g3_C10_13[6,6]=-1
