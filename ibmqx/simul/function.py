import numpy as np
from numpy import linalg as LA
np.set_printoptions(precision=7,suppress=True,linewidth=300)
from random import randint
import scipy as sy
from scipy import spatial as sp

# simulator for a qubit system with a starting qubit wavefunction in an appropriate basis
# paritcularly designed for a 3-rdm

# # # # # #
# orb    % bt % d# % bit % 3# % INDEX
#
# 000111 %  7 % d0 % 111 % 7  %  0
# 001011 % 11 % d1 % 110 % 6  %  1
# 010101 % 21 % d2 % 101 % 5  %  2 
# 011001 % 25 % d3 % 100 % 4  %  3
# 100110 % 38 % d4 % 011 % 3  %  4
# 101010 % 42 % d5 % 010 % 2  %  5
# 110100 % 52 % d6 % 001 % 1  %  6
# 111000 % 56 % d7 % 000 % 0  %  7
# 
# # # # # #

def bit_str():
    a = np.array([7,11,21,25,38,42,52,56])
    return a

def rdm(wf):
	#given the wavefunction, will compute the relevant 1-RDM
	rdm = np.zeros((6,6),dtype=np.complex_)
	rdm[0,0] = wf[4]*np.conj(wf[4])+wf[5]*np.conj(wf[5])+wf[6]*np.conj(wf[6])+wf[7]*np.conj(wf[7])
	rdm[1,1] = wf[2]*np.conj(wf[2])+wf[3]*np.conj(wf[3])+wf[6]*np.conj(wf[6])+wf[7]*np.conj(wf[7])
	rdm[2,2] = wf[1]*np.conj(wf[1])+wf[3]*np.conj(wf[3])+wf[5]*np.conj(wf[5])+wf[7]*np.conj(wf[7])
	rdm[3,3] = 1-rdm[2,2]
	rdm[4,4] = 1-rdm[1,1]
	rdm[5,5] = 1-rdm[0,0]
	rdm[0,5] = 1*(wf[0]*np.conj(wf[4])+wf[1]*np.conj(wf[5])+wf[2]*np.conj(wf[6])+wf[3]*np.conj(wf[7]))
	rdm[5,0] = np.conj(rdm[0,5])
	rdm[1,4] = -1*(wf[0]*np.conj(wf[2])+wf[1]*np.conj(wf[3])+wf[4]*np.conj(wf[6])+wf[5]*np.conj(wf[7]))
	rdm[4,1] = np.conj(rdm[1,4])
	rdm[2,3] = 1*(wf[0]*np.conj(wf[1])+wf[2]*np.conj(wf[3])+wf[4]*np.conj(wf[5])+wf[6]*np.conj(wf[7]))
	rdm[3,2] = np.conj(rdm[2,3])
	natocc, natorb = LA.eig(rdm)
	return rdm,natocc,natorb
	

def match(bit):
		# takes a bit, and outputs the index for the bitstr 
		if bit>30:
			if bit>50:
				if bit>54:
					new=7
				else:
					new=6
			else:
				if bit>40:
					new=5
				else:
					new=4
		else:
			if bit>20:
				if bit>23:
					new=3
				else:
					new=2
			else:
				if bit>10:
					new=1
				else:
					new=0
		return new




def exc_ij_str(det_bit,i,ival,j,jval): #returns a bit value for a string
	det_bit = det_bit[0:i] + str(ival) + det_bit[i+1:]
	det_bit = det_bit[0:j] + str(jval) + det_bit[j+1:]
	det_bit = det_bit[::-1]
	bit = 0
	for i in range(0,6):
		bit += int(det_bit[i])*2**i 
	bit = match(bit)
	return bit # as a number

def exc_ijkl_str(det_bit,i,ival,j,jval,k,kval,l,lval):
	det_bit = det_bit[0:i] + str(ival) + det_bit[i+1:]
	det_bit = det_bit[0:j] + str(jval) + det_bit[j+1:]
	det_bit = det_bit[0:k] + str(kval) + det_bit[k+1:]
	det_bit = det_bit[0:l] + str(lval) + det_bit[l+1:]
	det_bit = det_bit[::-1]	
	bit = 0
	for z in range(0,6):
		bit += int(det_bit[z])*(2**z) 
	bit = match(bit)
	return bit # as a number


def m_i(i,m): 
	m = np.asmatrix(m) #2x2 matrix
	new = np.identity(8,dtype=np.complex_)
	hold = bit_str()
	j = 5-i  #i and j are on the same qubit!!!!!! 
	for item in hold:	
		cur = match(item) #current bit
		a = format(item,'06b') #returns index as bit string, 6 items long 	
		#print(a, item, cur)
		if a[i]=='0': #then qubit in the 1 state, 
			h11 = exc_ij_str(a,i,0,j,1) #what we call the `stay put'...11
			h21 = exc_ij_str(a,i,1,j,0) #what we call the change from a+_i a_j, second column
			new[h11,cur] = m[1,1]
			new[h21,cur] = m[0,1]
		else:
			h12 = exc_ij_str(a,i,0,j,1) #  change
			h22 = exc_ij_str(a,i,1,j,0) # this is to stay put - maybe this is right
			new[h12,cur] = m[1,0]
			new[h22,cur] = m[0,0]
	return new

def norm(wavefunction):	
	summand = 0
	for i in range(0,8):
		summand += wavefunction[i]*np.conjugate(wavefunction[i])
	return summand


def m_ij(i,k,m):
	m = np.asmatrix(m) #2x2 matrix
	new = np.identity(8,dtype=np.complex_)
	hold = bit_str()
	j = 5-i #i,j, are same qubit
	l = 5-k #k, l are same qubit
	for item in hold:
		cur = match(item) #current bit
		a = format(item,'06b') #returns index as bit string, 6 items long 
		if (a[i]=='0' and a[k]=='0'): #ij, kl, qubit in 11 state
			h11 = exc_ijkl_str(a,i,0,j,1,k,0,l,1) # stay put...which 4,4 
			h21 = exc_ijkl_str(a,i,0,j,1,k,1,l,0)
			h31 = exc_ijkl_str(a,i,1,j,0,k,0,l,1)
			h41 = exc_ijkl_str(a,i,1,j,0,k,1,l,0)
			new[h11,cur] = m[3,3] 
			new[h21,cur] = m[2,3] 
			new[h31,cur] = m[1,3] 
			new[h41,cur] = m[0,3] 
		elif (a[i]=='0' and a[k]=='1'): # qubit in 10 state
			h12 = exc_ijkl_str(a,i,0,j,1,k,0,l,1)
			h22 = exc_ijkl_str(a,i,0,j,1,k,1,l,0)
			h32 = exc_ijkl_str(a,i,1,j,0,k,0,l,1)
			h42 = exc_ijkl_str(a,i,1,j,0,k,1,l,0)
			new[h12,cur] = m[3,2] 
			new[h22,cur] = m[2,2] 
			new[h32,cur] = m[1,2] 
			new[h42,cur] = m[0,2] 
		elif (a[i]=='1' and a[k]=='0'):   # qubit in 01 state
			h13 = exc_ijkl_str(a,i,0,j,1,k,0,l,1)
			h23 = exc_ijkl_str(a,i,0,j,1,k,1,l,0)
			h33 = exc_ijkl_str(a,i,1,j,0,k,0,l,1)
			h43 = exc_ijkl_str(a,i,1,j,0,k,1,l,0)
			new[h13,cur] = m[3,1] 
			new[h23,cur] = m[2,1] 
			new[h33,cur] = m[1,1] 
			new[h43,cur] = m[0,1] 
		elif (a[i]=='1' and a[k]=='1'):   #qubit in 00 state
			h14 = exc_ijkl_str(a,i,0,j,1,k,0,l,1)
			h24 = exc_ijkl_str(a,i,0,j,1,k,1,l,0)
			h34 = exc_ijkl_str(a,i,1,j,0,k,0,l,1)
			h44 = exc_ijkl_str(a,i,1,j,0,k,1,l,0)
			new[h14,cur] = m[3,0] 
			new[h24,cur] = m[2,0] 
			new[h34,cur] = m[1,0] 
			new[h44,cur] = m[0,0]
		else:
			print('Error!')
	return new

def mml(mset,wave):
	for matrix in mset:
		wave = np.matmul(matrix,wave)
	return wave

def addToFile(file, what):
	f = open(file, 'a+')
	f.write(what) 
	f.close()


def saveocc(savefile,occ):
	f = savefile
	a = '  '.join('%0.5f' %x for x in occ)
	f.write(a)
	f.write('\n')

def saveocc2(file,occ):
	f = open(file,'a+')
	f.write('  '.join('%0.5f' %x for x in occ))
	f.write('\n')
	f.close()


##################

def rot(theta):
	matr = np.matrix([[np.cos(np.deg2rad(theta)),-np.sin(np.deg2rad(theta))],[np.sin(np.deg2rad(theta)),np.cos(np.deg2rad(theta))]])
	return matr

def rot_rand():
	theta = randint(-90,90)
	matr = np.matrix([[np.cos(np.deg2rad(theta)),-np.sin(np.deg2rad(theta))],[np.sin(np.deg2rad(theta)),np.cos(np.deg2rad(theta))]])
	return matr

rt = 1/np.sqrt(2)
H = np.matrix([[rt,rt],[rt,-rt]])
Hm = np.matrix([[rt,-rt],[-rt,-rt]])
Z = np.matrix([[1,0],[0,-1]])
X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])

CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
C_ph = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1.j]])
C_z = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])

rCNOT = np.matrix([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
##################
def order(i):
	hold = np.matrix([[0,2,0,1,1,2],[0,2,0,1,2,1],[0,2,1,0,1,2],[0,2,1,0,2,1],[2,0,0,1,1,2],[2,0,0,1,2,1],[2,0,1,0,1,2],[2,0,1,0,2,1]])
	return hold[i,:]

def rdm_rot(p1,p2,p3,p4,p5,p6,oldord,ut): #
	tf = []
	ords = order(oldord).tolist()[0]
	nocc = []
	tf.append(m_i(ords[0],rot(p1)))
	tf.append(m_i(ords[1],rot(p2)))	
	tf.append(m_ij(ords[0],ords[1],ut[0])) #CLASS 1
	tf.append(m_i(ords[2],rot(p3)))
	tf.append(m_i(ords[3],rot(p4)))	
	tf.append(m_ij(ords[2],ords[3],ut[1])) #CLASS 1
	tf.append(m_i(ords[4],rot(p5)))
	tf.append(m_i(ords[5],rot(p6)))	
	tf.append(m_ij(ords[4],ords[5],ut[2])) #CLASS 1
	wf = np.array([0,0,0,0,0,0,0,1])
	wf = mml(tf,wf)
	nrdm,natocc,natorb = rdm(wf)
	natocc = natocc.real
	return natocc

def unpack(var,arg):
	p1,p2,p3,p4,p5,p6 = var[:]	
	pt,ord,ut = arg[0],arg[1],arg[2:]
	occnum = rdm_rot(p1,p2,p3,p4,p5,p6,ord,ut)
	return occnum 

def f1(var,arg): 	
	p1,p2,p3,p4,p5,p6 = var[:]	
	pt,ord,ut = arg[0],arg[1],arg[2:]
	occnum = rdm_rot(p1,p2,p3,p4,p5,p6,ord,ut)
	occnum.sort()
	dist = sp.distance.euclidean(pt,occnum[-1:-4:-1])		
	#print("%0.7f" % dist)
	return dist


def measure(wf):
    # returns the qubit state of a wavefunction
    occ = np.zeros(6)
    occ[0] = wf[4]*np.conj(wf[4])+wf[5]*np.conj(wf[5])+wf[6]*np.conj(wf[6])+wf[7]*np.conj(wf[7])
    occ[1] = wf[2]*np.conj(wf[2])+wf[3]*np.conj(wf[3])+wf[6]*np.conj(wf[6])+wf[7]*np.conj(wf[7])
    occ[2] = wf[1]*np.conj(wf[1])+wf[3]*np.conj(wf[3])+wf[5]*np.conj(wf[5])+wf[7]*np.conj(wf[7])
    occ[3] = 1-occ[2]
    occ[4] = 1-occ[1]
    occ[5] = 1-occ[0]
    return occ
