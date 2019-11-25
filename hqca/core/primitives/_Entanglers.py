from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit

def _ent1_Ry_cN(qgdc,phi,i,k,ddphi=False,**kw):
    if not ddphi:
        qgdc.qc.cx(self.q[k],self.q[i])
        qgdc.qc.x(self.q[k])
        qgdc.qc.ry(phi/2,self.q[k])
        qgdc.qc.cx(self.q[i],self.q[k])
        qgdc.qc.ry(-phi/2,self.q[k])
        qgdc.qc.cx(self.q[i],self.q[k])
        qgdc.qc.x(self.q[k])
        qgdc.qc.cx(self.q[k],self.q[i])
        #for s in range(i,k):
            #qgdc.qc.cz(self.q[k],self.q[s])
            #qgdc.qc.z(self.q[s])

def _Uent1_cN(qgdc,phi1,phi2,i,k,**kw):
    qgdc.qc.cx(self.q[k],self.q[i])
    qgdc.qc.x(self.q[k])
    qgdc.qc.rz(phi1,self.q[k])
    qgdc.qc.ry(-phi2,self.q[k])
    qgdc.qc.cz(self.q[i],self.q[k])
    qgdc.qc.ry(phi2,self.q[k])
    qgdc.qc.rz(-phi1,self.q[k])
    qgdc.qc.x(self.q[k])
    qgdc.qc.cx(self.q[k],self.q[i])

def _phase(qgdc,phi,theta,i,k,**kw):
    qgdc.qc.rz(phi,self.q[i])
    qgdc.qc.rz(theta,self.q[k])
