import numpy as np
try: 
    from . import function as fx
except:
    import function as fx
np.set_printoptions(suppress=True,precision=4)
from random import randint
from numpy import linalg as LA

# contains functions for running the sesmart simulator

def debug(print_string,dbg=False):
    if dbg:
        print(print_string)
    return 

def single_run_c3(theta1,theta2,theta3,order,dbg=False):
    tf = []
    wf = np.array([0,0,0,0,0,0,0,1])
    tf.append(fx.m_i(order[0],fx.rot(theta1)))
    tf.append(fx.m_ij(order[0],order[1],fx.CNOT)) #CLASS 1	
    tf.append(fx.m_i(order[2],fx.rot(theta2)))
    tf.append(fx.m_ij(order[2],order[3],fx.CNOT)) #CLASS 2	
    tf.append(fx.m_i(order[4],fx.rot(theta3)))
    tf.append(fx.m_ij(order[4],order[5],fx.CNOT)) #CLASS 3	
    wf = fx.mml(tf,wf)
    return wf

def single_run_c2(theta1,theta2,order,dbg=False):
    tf = []
    wf = np.array([0,0,0,0,0,0,0,1])
    tf.append(fx.m_i(order[0],fx.rot(theta1)))
    tf.append(fx.m_ij(order[0],order[1],fx.CNOT)) #CLASS 1	
    tf.append(fx.m_i(order[2],fx.rot(theta2)))
    tf.append(fx.m_ij(order[2],order[3],fx.CNOT)) #CLASS 2	
    wf = fx.mml(tf,wf)
    return wf



def single_full_run_c3(theta1,theta2,theta3,theta4,theta5,theta6,order,dbg=False):
    tf = []
    wf = np.array([0,0,0,0,0,0,0,1])
    tf.append(fx.m_i(order[0],fx.rot(theta1)))
    tf.append(fx.m_i(order[1],fx.rot(theta2)))
    tf.append(fx.m_ij(order[0],order[1],fx.CNOT)) #CLASS 1	
    tf.append(fx.m_i(order[2],fx.rot(theta3)))
    tf.append(fx.m_i(order[3],fx.rot(theta4)))
    tf.append(fx.m_ij(order[2],order[3],fx.CNOT)) #CLASS 2	
    tf.append(fx.m_i(order[4],fx.rot(theta5)))
    tf.append(fx.m_i(order[5],fx.rot(theta6)))
    tf.append(fx.m_ij(order[4],order[5],fx.CNOT)) #CLASS 3	
    wf = fx.mml(tf,wf)
    return wf


def single_full_run_c2(theta1,theta2,theta3,theta4,order,dbg=False):
    tf = []
    wf = np.array([0,0,0,0,0,0,0,1])
    tf.append(fx.m_i(order[0],fx.rot(theta1)))
    tf.append(fx.m_i(order[1],fx.rot(theta2)))
    tf.append(fx.m_ij(order[0],order[1],fx.CNOT)) #CLASS 1	
    tf.append(fx.m_i(order[2],fx.rot(theta3)))
    tf.append(fx.m_i(order[3],fx.rot(theta4)))
    tf.append(fx.m_ij(order[2],order[3],fx.CNOT)) #CLASS 2	
    wf = fx.mml(tf,wf)
    return wf

def single_full_run_c1(theta1,theta2,order,dbg=False):
    tf = []
    wf = np.array([0,0,0,0,0,0,0,1])
    tf.append(fx.m_i(order[0],fx.rot(theta1)))
    tf.append(fx.m_i(order[1],fx.rot(theta2)))
    tf.append(fx.m_ij(order[0],order[1],fx.CNOT)) #CLASS 1	
    wf = fx.mml(tf,wf)
    return wf

def construct_rdm(wf,dbg=False):
    diag = fx.measure(wf)
    debug('Here is the wavefunction and the measurement of the wavefunction:',dbg)
    debug(wf,dbg)
    debug(diag,dbg)
    wf = fx.mml([fx.m_i(0,fx.rot(+45))],wf)
    debug('\nWavefunction after O_R1',dbg)
    debug(wf,dbg)
    wf = fx.mml([fx.m_i(1,fx.rot(-45))],wf)
    debug('Wavefunction after O_R2',dbg)
    debug(wf,dbg)
    #debug(fx.m_i(1,fx.rot(-45)),dbg)
    wf = fx.mml([fx.m_i(2,fx.rot(+45))],wf)
    debug('Wavefunction after O_R3',dbg)
    debug(wf,dbg)
    off_diag = fx.measure(wf)
    debug('\nHere is the wavefunction and the measurement of the off-diagonal:',dbg)
    debug(wf,dbg)
    debug(off_diag,dbg)
    rdm = np.zeros((6,6))
    for i in range(0,3):
        rdm[i,i] = diag[i]
        rdm[5-i,5-i] = 1-rdm[i,i]
        if i==1:
            rdm[5-i,i] = +off_diag[i]-0.5
        else:
            rdm[5-i,i] = -off_diag[i] +0.5
        rdm[i,5-i] = rdm[5-i,i]
    return rdm

def verify(wf,err_thrsh):
    rdm,nocc,norb = fx.rdm(wf)
    rdm_test = construct_rdm(wf)
    diff = rdm - rdm_test
    for r in range(0,6):
        for c in range(0,6):
            if abs(diff[r,c])>err_thrsh:
                use = False
                return diff, use
            use=True
    return nocc,use

   
def debug_run(the1,the2,the3,order,thrsh):
    
    wf = single_run_c3(the1,the2,the3,order)
    print(np.asmatrix(wf).T)
    data,use = verify(wf,thrsh) 


    rdm = construct_rdm(wf,dbg=False)

    nocc, norb =  LA.eig(rdm)
    rdm,nocc,norb = fx.rdm(wf)
    return print('Done')

def euclid(point_A,point_B):
    point_A = np.asarray(point_A)
    point_B = np.asarray(point_B)
    return np.sqrt(np.sum(np.square(point_A-point_B)))
        
def nearest_neighbor(point,set_of_points,set_nn=False):
    nearest_distance = np.sqrt(3/8)
    if set_nn==False:
        for test_point in set_of_points:
            dist = euclid(point,test_point)
            if dist<nearest_distance:
                nearest_distance = dist
            else:
                pass
    else:
        nearest_distance = np.ones(set_nn)*nearest_distance
        for test_point in set_of_points:
            dist = euclid(point,test_point)
            if dist<nearest_distance[0]:
                i = set_nn
                for i in reversed(range(1,set_nn)):
                    nearest_distance[i] = nearest_distance[i-1]
                nearest_distance[0] = dist
            else:
                pass
    return nearest_distance



def multi_run_c3(the1,the2,the3,order,thrsh,rad=True):
    if rad==True:
        the1*= 180/np.pi
        the2*= 180/np.pi
        the3*= 180/np.pi
    size = len(the1)*len(the2)*len(the3)*len(order) 
    hold = np.zeros((size,6))
    ind = 0
    for one in the1:
        for two in the2:
            for thr in the3:
                for ords in order:
                    wf = single_run_c3(one,two,thr,ords)
                    data,use = verify(wf,thrsh) 
                    if not use:
                        print('Error in calculation. Check quantum simulator.') 
                        print('Difference in RDM:', '\n',diff)
                        hold = hold[0:ind,:]
                        err=True
                        return hold, err
                        
                    else:
                        data.sort()
                        hold[ind,:] = data
                    ind+=1
    if rad==True:
        the1*= np.pi/180
        the2*= np.pi/180
        the3*= np.pi/180
    err=False
    return hold , err

'''
###
t1 = 60
t2 = 30
t3 = 15
#t1 = 45
#t2 = 60
#t3 = 15
print(t1,t2,t3)
tf = [[]]
tf[0].append(fx.m_i(0,fx.rot(t1)))
#print(fx.m_i(0,fx.rot(t1)))
tf[0].append(fx.m_ij(0,2,fx.CNOT)) #CLASS 1	
tf[0].append(fx.m_i(0,fx.rot(t2)))
tf[0].append(fx.m_ij(0,1,fx.CNOT)) #CLASS 2	
tf[0].append(fx.m_i(2,fx.rot(t3)))
tf[0].append(fx.m_ij(2,1,fx.CNOT)) #CLASS 3	
wf = np.array([0,0,0,0,0,0,0,1])


wf = fx.mml(tf[0],wf)
###

#wf = mm(C_ph_12,wf)
#wf = mm(CNOT_02,wf)
#wf = mm(H_1,wf)


#print(CNOT_12)
print(wf)
a,b,c = fx.rdm(wf)
print(a)
print(b)
#print(fx.norm(wf))
#print(c)

zeta = fx.measure(wf)
test = np.square(wf)
index =  ['111','110','101','100','011','010','001','000']
data = zip(index,test)
print(data)
wf = fx.mml([fx.m_i(2,fx.rot(-45))],wf)
#print(wf)
zed = fx.measure(wf)
offdiag = (zed[2]-0.5)
#print(a[3,2])
#print(offdiag)


'''
