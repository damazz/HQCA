

def h2_hamiltonian(Q,zi,iz,zz,xx,scaling=1.0):
    Q.qc.rz(zi*scaling,Q.q[0])
    Q.qc.rz(iz*scaling,Q.q[1])
    Q.qc.cx(Q.q[0],Q.q[1])
    Q.qc.rx(xx*scaling,Q.q[0])
    Q.qc.rz(zz*scaling,Q.q[1])
    Q.qc.cx(Q.q[0],Q.q[1])

def xy_yx_ih_simple(Q,xy,yx,zi,iz,zz,xx,scaling=1.0):
    Q.qc.ry(xy+yx,Q.q[0])
    Q.qc.rx(xx*scaling,Q.q[0])
    Q.qc.rz(zi*scaling,Q.q[0])
    Q.qc.rz(zz*scaling,Q.q[1])
    Q.qc.cx(Q.q[0],Q.q[1])
    Q.qc.rz(iz*scaling,Q.q[1])

def xy_yx_simple(Q,xy,yx):
    Q.qc.ry(xy+yx,Q.q[0])
    Q.qc.cx(Q.q[0],Q.q[1])


def xy_yx_gate(Q,xy,yx,v2=True):
    if v2:
        Q.qc.ry(xy,Q.q[0])
        Q.qc.cx(Q.q[1],Q.q[0])
        Q.qc.ry(xy,Q.q[0])
        Q.qc.cx(Q.q[1],Q.q[0])
    else:
        #Q.qc.cx(Q.q[1],Q.q[0])
        Q.qc.ry(xy,Q.q[1])
        Q.qc.cz(Q.q[1],Q.q[0])
        Q.qc.ry(yx,Q.q[1])

        Q.qc.s(Q.q[0])
        Q.qc.z(Q.q[0])
        Q.qc.cx(Q.q[1],Q.q[0])
        Q.qc.s(Q.q[0])
        Q.qc.z(Q.q[1])
        Q.qc.s(Q.q[1])




