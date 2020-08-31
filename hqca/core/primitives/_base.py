

def apply_cx(Q,i,j,**kw):
    Q.qc.cx(Q.q[i],Q.q[j])

def apply_cz(Q,i,j,**kw):
    Q.qc.cz(Q.q[i],Q.q[j])

def apply_si(Q,i,**kw):
    Q.qc.sdg(Q.q[i])

def apply_s(Q,i,**kw):
    Q.qc.s(Q.q[i])

def apply_h(Q,i,**kw):
    Q.qc.h(Q.q[i])

def apply_rz(Q,i,theta):
    Q.qc.rz(theta,Q.q[i])

def apply_ry(Q,i,theta):
    Q.qc.ry(theta,Q.q[i])

def apply_rx(Q,i,theta):
    Q.qc.rx(theta,Q.q[i])

def apply_x(Q,i):
    Q.qc.x(Q.q[i])

def apply_z(Q,i):
    Q.qc.z(Q.q[i])

def apply_y(Q,i):
    Q.qc.y(Q.q[i])