import sys
sys.path+= ['/home/scott/Documents/research/3_vqa']
import hqca.tools.RDMFunctions as rdmf
import hqca.ibmqx.qfunc.rdm as rdm
from qiskit import QuantumProgram, QuantumCircuit
import numpy as np

class SQP:
    def __init__(self,backend_input='local_qasm_simulator'):
        self.qp = QuantumProgram()
        self.backend=backend_input
        self.circuits = []
    def ry6p(self,parameters,specify_circuit,tomo_target,qb_parity):
        self.qorder = [0,2,0,1,2,1]
        self.circuits.append('{}{}'.format(specify_circuit,tomo_target))
        para = []
        for par in parameters:
            para.append(par*np.pi/180)
        self.qreg = self.qp.create_quantum_register('{}q{}qr'.format(specify_circuit,tomo_target),3)
        self.creg = self.qp.create_classical_register('{}q{}cr'.format(specify_circuit,tomo_target),3)
        self.circ = self.qp.create_circuit('{}{}'.format(specify_circuit,tomo_target),[self.qreg],[self.creg])
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[1]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]]) 
        self.circ.ry(2*para[2],self.qreg[self.qorder[2]])
        self.circ.ry(2*para[3],self.qreg[self.qorder[3]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]]) 
        self.circ.ry(2*para[4], self.qreg[self.qorder[4]])
        self.circ.ry(2*para[5], self.qreg[self.qorder[5]])
        self.circ.cx(self.qreg[self.qorder[4]],self.qreg[self.qorder[5]]) 
        if specify_circuit=='ij':
            self.circ.h(self.qreg[0])
            self.circ.z(self.qreg[0])
            self.circ.z(self.qreg[1])
            self.circ.h(self.qreg[1])
            self.circ.h(self.qreg[2])
            self.circ.z(self.qreg[2])
            #self.circ.ry(np.pi/2,self.qreg[0])
            #self.circ.ry(-np.pi/2,self.qreg[1])
            #self.circ.ry(np.pi/2,self.qreg[2])
        elif specify_circuit=='iklj':
            ind1 = int(tomo_target[0])
            ind2 = int(tomo_target[1])
            s1 = qb_parity[ind1]
            s2 = qb_parity[ind2]
            self.circ.cx(self.qreg[ind1],self.qreg[ind2])
            if s1*s2==1:
                self.circ.h(self.qreg[ind1])
                self.circ.z(self.qreg[ind1])
            else:
                self.circ.z(self.qreg[ind1])
                self.circ.h(self.qreg[ind1])
            self.circ.cx(self.qreg[ind1],self.qreg[ind2])
        elif specify_circuit=='iklj2':
            ind1 = int(tomo_target[0])
            ind2 = int(tomo_target[1])
            s1 = qb_parity[ind1]
            s2 = qb_parity[ind2]
            self.circ.cx(self.qreg[ind2],self.qreg[ind1])
            if s1*s2==1:
                self.circ.h(self.qreg[ind2])
                self.circ.z(self.qreg[ind2])
            else:
                self.circ.z(self.qreg[ind2])
                self.circ.h(self.qreg[ind2])
            self.circ.cx(self.qreg[ind2],self.qreg[ind1])
            '''
        elif specify_circuit=='ikli': #entangles + rotations
            ind1 = int(tomo_target[0])
            ind2 = int(tomo_target[1])
            ind3 = int(tomo_target[2])
            s1 = qb_parity[ind1]
            s2 = qb_parity[ind2]
            s3 = qb_parity[ind3]
            self.circ.cz(self.qreg[ind1],self.qreg[ind2])
            if s1*s2==1:
                self.circ.h(self.qreg[ind2])
                self.circ.z(self.qreg[ind2])
            else:
                self.circ.z(self.qreg[ind2])
                self.circ.h(self.qreg[ind2])
            self.circ.cz(self.qreg[ind1],self.qreg[ind2])

            self.circ.cz(self.qreg[ind1],self.qreg[ind3])
            if s1*s3==1:
                self.circ.h(self.qreg[ind3])
                self.circ.z(self.qreg[ind3])
            else:
                self.circ.z(self.qreg[ind3])
                self.circ.h(self.qreg[ind3])
            self.circ.cz(self.qreg[ind1],self.qreg[ind3])

            if s1==-1:
                self.circ.z(self.qreg[ind1])
                self.circ.h(self.qreg[ind1])
            else:
                self.circ.h(self.qreg[ind1])
                self.circ.z(self.qreg[ind1])
            '''
        elif specify_circuit=='jlkj': #entangles + rotations
            ind1 = int(tomo_target[0])
            ind2 = int(tomo_target[1])
            ind3 = int(tomo_target[2])
            s1 = qb_parity[ind1]
            s2 = qb_parity[ind2]
            s3 = qb_parity[ind3]
            self.circ.cz(self.qreg[ind1],self.qreg[ind2])
            if s2==-1:
                self.circ.h(self.qreg[ind2])
                self.circ.z(self.qreg[ind2])
            else:
                self.circ.z(self.qreg[ind2])
                self.circ.h(self.qreg[ind2])

            self.circ.cz(self.qreg[ind1],self.qreg[ind2])
            if s3==1:
                self.circ.z(self.qreg[ind3])
                self.circ.h(self.qreg[ind3])
            else:
                self.circ.h(self.qreg[ind3])
                self.circ.z(self.qreg[ind3])

        elif specify_circuit=='ikki':
            ind1 = int(tomo_target[0])
            ind2 = int(tomo_target[1])
            s1 = qb_parity[ind1]
            s2 = qb_parity[ind2]
            if s1*s2==1:
                self.circ.cx(self.qreg[ind1],self.qreg[ind2])
            else:
                self.circ.z(self.qreg[ind1])
                self.circ.cx(self.qreg[ind1],self.qreg[ind2])
        self.circ.measure(self.qreg[0],self.creg[0])
        self.circ.measure(self.qreg[1],self.creg[1])
        self.circ.measure(self.qreg[2],self.creg[2])

    def execute(self,num_shots,nmeasure):
        self.qobj = self.qp.compile(self.circuits,shots=num_shots,backend=self.backend)
        self.results = self.qp.run(self.qobj)
        return self.results,self.circuits

    def calc_ideal(self):
        self.unit_results = self.qp.execute('ii',backend='local_unitary_simulator')
        self.unit_transformation = self.unit_results.get_data('ii')['unitary']
        self.unit_results_err = self.qp.execute('ij',backend='local_unitary_simulator')
        self.unit_transformation_err = self.unit_results_err.get_data('ij')['unitary']
        self.unit_state = self.unit_transformation[:,0]

    def get_ideal(self):
        self.calc_ideal()
        return self.unit_transformation,self.unit_transformation_err

def rdm(data,unitary=False):
    #takes in a set of counts data in the dictionary form, decodes it, and 
    # obtains the RDM element 
    if unitary==True:
        wf = np.matrix([[1],[0],[0],[0],[0],[0],[0],[0]])
        wf = data*wf
        wf = (np.real(np.round(np.square(wf)*1000000)))
        wf = [int(wf[i,0]) for i in range(0,8)]
        #print(wf)
        unit = ['000','001','010','011','100','101','110','111']
        data = dict(zip(unit,wf))
        print(data)
    else:
        unit = list(data.keys())
        #print(unit)
    r = np.zeros(len(unit[0]))
    total_count = 0
    for qubit, count in data.items():
        total_count += count
        n_qb = len(qubit)
        for i in range(0,n_qb):
            if qubit[n_qb-1-i]=='0':
                r[n_qb-1-i]+= count
    r = np.multiply(r,total_count**-1) 
    return r


def construct_rdm(diag,rot):
    rdm = np.zeros((6,6))
    for i in range(0,len(rot)):
        rdm[5-i,i] = rot[i]-0.5
        rdm[i,i] = diag[i]
        if i==1:
            rdm[5-1,i]*= -1
        rdm[i,5-i] = rdm[5-i,i]
        rdm[5-i,5-i] = 1 - rdm[i,i]
    evalues,evector = np.linalg.eig(rdm)
    return rdm,evalues,evector

def counts_to_1rdm(main,err,backend='local_qasm_simulator',use_err=True,unitary=False):
    if backend=='ibmqx2':
        trace=[3,4]
    elif backend=='local_qasm_simulator':
        trace=[]
    else:
        trace=[]
    if use_err:
        ONrdm, ON, ONvec = construct_rdm(
            rdm(main,unitary
                ),
            rdm(err,unitary
                )
            )
    else:
        ONrdm, ON, ONvec = construct_rdm(
            rdm.rdm(
                rdm.filt(main,trace)),
                [0.5,0.5,0.5]
                )
    return ON,ONrdm

def tomography(parameters,backend,shots=1024,rdm1=True,rdm2=True,unitary=False):
    qp = SQP(backend)
    # 1-RDM
    if rdm1:
        qb_par = {0:1,1:-1,2:1}
        qp.ry6p(parameters,'ii','',qb_par)
        qp.ry6p(parameters,'ij','',qb_par)
        res, circ = qp.execute(shots,[0,1,2])
        return res, circ
    if rdm2:
        qb_par = {0:1,1:-1,2:1}
        qp.ry6p(parameters,'iklj','01',qb_par)
        qp.ry6p(parameters,'iklj','12',qb_par)
        qp.ry6p(parameters,'iklj','20',qb_par)

        #qp.ry6p(parameters,'ikli','210',qb_par)
        #qp.ry6p(parameters,'ikli','102',qb_par)
        #qp.ry6p(parameters,'ikli','021',qb_par)

        qp.ry6p(parameters,'jlkj','021',qb_par)
        qp.ry6p(parameters,'jlkj','201',qb_par)
        qp.ry6p(parameters,'jlkj','120',qb_par)
        qp.ry6p(parameters,'jlkj','210',qb_par)
        qp.ry6p(parameters,'jlkj','012',qb_par)
        qp.ry6p(parameters,'jlkj','102',qb_par)

        qp.ry6p(parameters,'ikki','01' ,qb_par)
        qp.ry6p(parameters,'ikki','12' ,qb_par)
        qp.ry6p(parameters,'ikki','20' ,qb_par)
        res, circ = qp.execute(shots,[0,1,2])
        return res, circ
    if unitary:
        qb_par = {0:1,1:-1,2:1}
        qp.ry6p(parameters,'ii','',qb_par)
        qp.ry6p(parameters,'ij','',qb_par)
        res = qp.get_ideal()
        circ = qp.circuits
        return res, circ


def measure(data,reverse=False):
    unit = list(data.keys())
    total_count=0
    r = np.zeros(len(unit[0]))
    for qubit, count in data.items():
        total_count += count
        n_qb = len(qubit)
        for i in range(0,n_qb):
            if qubit[n_qb-1-i]=='0':
                if reverse:
                    r[i]+= count
                else:
                    r[n_qb-1-i]+= count
    r = np.multiply(r,total_count**-1)
    return r # len in how many qubits

def qb(r):
    #return abs(2-r)
    return r

def assemble_2rdm(results,circuits,qb_parity):
    rdm2 = np.zeros((6,6,6,6))
    Lc = len(circuits)
    ##### iklj ##### 
    rhold={}
    ihold={}
    for item in circuits:
        if item[0:4]=='iklj':
            q1 = qb(int(item[4]))
            q2 = qb(int(item[5]))
            temp = measure(results.get_counts(item),reverse=True)
            alp1 = 1 - 2*temp[q1] # sum of terms 
            alp2 = 1 - 2*temp[q2] # difference of terms 
            # check for fermionic sign here
            temp1 = 0.25*(alp1+alp2)
            temp2 = 0.25*(alp1-alp2)
            q1 = qb(q1)
            q2 = qb(q2)
            #i = 5-q1
            #j = q1
            #k = 5-q2
            #l = q2
            i = 3+q1
            j = 2-q1
            k = 3+q2
            l = 2-q2
            s1 = qb_parity[j]
            s2 = qb_parity[l]
            rdm2[i,k,l,j]+= temp1*s1*s2
            rdm2[k,i,j,l]+= temp1*s1*s2
            rdm2[i,k,j,l]-= temp1*s1*s2
            rdm2[k,i,l,j]-= temp1*s1*s2

            rdm2[j,l,k,i]+= temp1*s1*s2
            rdm2[l,j,i,k]+= temp1*s1*s2
            rdm2[l,j,k,i]-= temp1*s1*s2
            rdm2[j,l,i,k]-= temp1*s1*s2

            rdm2[i,l,k,j]+= temp2*s1*s2
            rdm2[l,i,j,k]+= temp2*s1*s2
            rdm2[l,i,k,j]-= temp2*s1*s2
            rdm2[i,l,j,k]-= temp2*s1*s2

            rdm2[j,k,l,i]+= temp2*s1*s2
            rdm2[k,j,i,l]+= temp2*s1*s2
            rdm2[k,j,l,i]-= temp2*s1*s2
            rdm2[j,k,i,l]-= temp2*s1*s2

            ###
        if item[0:4]=='ikli':
            q1 = item[4]
            q2 = item[5]
            q3 = item[6]
            rhold[q1+q2]={}
            rhold[q1+q3]={}
        if item[0:4]=='jlkj':
            q1 = item[4]
            q2 = item[5]
            q3 = item[6]
            rhold[q1+q2]={}
        if item[0:4]=='ikki':
            ihold[item[4:6]]={}
    for item in circuits:
        if item[0:4]=='ikli':
            q1 = (int(item[4]))
            q2 = (int(item[5]))
            q3 = (int(item[6]))
            tempc = measure(results.get_counts(item),reverse=True)
            rhold[str(q1)+str(q2)]['ikli']=tempc[q2]
            rhold[str(q2)+str(q1)]['ij']=tempc[q1]
            rhold[str(q1)+str(q3)]['ikli']=tempc[q3]
            rhold[str(q3)+str(q1)]['ij']=tempc[q1]
        elif item[0:4]=='jlkj':
            q1 = (int(item[4]))
            q2 = (int(item[5]))
            q3 = (int(item[6]))
            tempc = measure(results.get_counts(item),reverse=True)
            rhold[str(q1)+str(q2)]['ikli']=tempc[q2]
            rhold[str(q1)+str(q3)]['ij']=tempc[q3]
        elif item[0:4]=='ikki':
            q1 = qb(int(item[4]))
            q2 = qb(int(item[5]))
            qb_set = [0,1,2]
            qb_set.remove(q1)
            qb_set.remove(q2)
            q3 = qb_set[0]
            tempc = measure(results.get_counts(item),reverse=True)
            ihold[item[4:6]]['jllj']=tempc[q1]
            ihold[item[4:6]]['mc']=tempc[q2]
            if q3==0:
                new='20'
            elif q3==1:
                new='01'
            elif q3==2:
                new='12'
            ihold[new]['jj']=tempc[q3]

    for key, val in rhold.items():
        q1 = int(key[0])
        q2 = int(key[1])
        qb_set = [0,1,2]
        qb_set.remove(q1)
        qb_set.remove(q2)
        q3 = qb_set[0]

        bet1 = 1-2*val['ij']
        bet2 = 1-2*val['ikli']
        temp1 = 0.25*(bet1+bet2)
        temp2 = 0.25*(bet1-bet2)
        i = 3+q1
        j = 2-q1
        k = 3+q2
        l = 2-q2
        s1 = qb_parity[l] #note...this is on qubit 2
        print(key,val)
        print(temp1,temp2,i,k,l,j,s1)
        #s1 = qb_parity[j]
        rdm2[i,k,l,i]+= temp1*s1
        rdm2[k,i,i,l]+= temp1*s1
        rdm2[k,i,l,i]-= temp1*s1
        rdm2[i,k,i,l]-= temp1*s1

        rdm2[i,l,k,i]+= temp1*s1
        rdm2[l,i,i,k]+= temp1*s1
        rdm2[l,i,k,i]-= temp1*s1
        rdm2[i,l,i,k]-= temp1*s1

        rdm2[j,k,l,j]+= temp2*s1
        rdm2[k,j,j,l]+= temp2*s1
        rdm2[j,k,j,l]-= temp2*s1
        rdm2[k,j,l,j]-= temp2*s1

        rdm2[j,l,k,j]+= temp2*s1
        rdm2[l,j,j,k]+= temp2*s1
        rdm2[j,l,j,k]-= temp2*s1
        rdm2[l,j,k,j]-= temp2*s1

    for key, val in ihold.items():
        q1 = int(key[0])
        q2 = int(key[1])
        qb_set = [0,1,2]
        qb_set.remove(q1)
        qb_set.remove(q2)
        q3 = qb_set[0]
        q1 = qb(q1)
        q2 = qb(q2)

        #i = 5-q1
        #j = q1
        #k = 5-q2
        #l = q2
        i = 3+q1
        j = 2-q1
        k = 3+q2
        l = 2-q2
        s1 = qb_parity[2-q1]
        s2 = qb_parity[2-q2]

        m_j = val['jllj']
        m_c = val['mc']
        m_l = val['jj']
        m_k = 1-m_l
        m_i = 1-m_j

        temp1 = 0.5*(m_i+(m_c-m_l)) #ikki
        temp2 = 0.5*(m_i-(m_c-m_l)) #illi
        temp3 = m_k-temp1 #jkkj
        temp4 = m_l-temp2 #jllj
        #if j==0 or l==0:
        #    #print(temp1,temp2,temp3,temp4)
        #    print(i,j,k,l)
        rdm2[j,l,l,j]+= temp4
        rdm2[l,j,j,l]+= temp4
        rdm2[j,l,j,l]-= temp4
        rdm2[l,j,l,j]-= temp4

        rdm2[j,k,k,j]+= temp3
        rdm2[k,j,j,k]+= temp3
        rdm2[k,j,k,j]-= temp3
        rdm2[j,k,j,k]-= temp3

        rdm2[i,l,l,i]+= temp2
        rdm2[l,i,i,l]+= temp2
        rdm2[i,l,i,l]-= temp2
        rdm2[l,i,l,i]-= temp2

        rdm2[i,k,k,i]+= temp1
        rdm2[k,i,i,k]+= temp1
        rdm2[k,i,k,i]-= temp1
        rdm2[i,k,i,k]-= temp1

    return rdm2
