'''
hqca/quantum/algorithms/_ECC.py

has some error correction techniques

'''

def _ec_ucc2_parity_single(dc,i,j,k,l,an):
    dc.qc.cx(dc.q[i],dc.q[an])
    dc.qc.cx(dc.q[j],dc.q[an])
    dc.qc.cx(dc.q[k],dc.q[an])
    dc.qc.cx(dc.q[l],dc.q[an])

