'''
hqca/quantum/algorithms/_ECC.py

has some error correction techniques

'''

def _ec_ucc2_parity_single(dc,i,j,k,l,an):
    dc.qc.cx(dc.q[i],dc.q[an])
    dc.qc.cx(dc.q[j],dc.q[an])
    dc.qc.cx(dc.q[k],dc.q[an])
    dc.qc.cx(dc.q[l],dc.q[an])

def _ec_spin_parity(dc,i,j,k,l,an1,an2):
    dc.qc.cx(dc.q[i],dc.q[an1])
    dc.qc.cx(dc.q[j],dc.q[an1])
    dc.qc.cx(dc.q[k],dc.q[an2])
    dc.qc.cx(dc.q[l],dc.q[an2])


