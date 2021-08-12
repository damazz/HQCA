import numpy as np

def get_Q_matrix(rdm):
    if not self.p==2:
        sys.exit('Not order 2 g matrix.')
    if not self.dqg=='d':
        sys.exit('Getting Q from non-2RDM!.')
    self.expand()
    new = np.zeros(self.rdm.shape,dtype=np.complex_)
    d1 = self.reduce_order()
    for i in range(self.r):
        for j in range(self.r):
            ij = (i==j)
            for k in range(self.r):
                ik,jk = (i==k),(j==k)
                for l in range(self.r):
                    il,jl,kl = (i==l),(j==l),(k==l)
                    new[j,l,i,k]+= self.rdm[i,k,j,l]
                    new[j,l,i,k]+= -il*jk
                    new[j,l,i,k]+= il*d1.rdm[k,j]
                    new[j,l,i,k]+= -kl*d1.rdm[i,j]
                    new[j,l,i,k]+= jk*d1.rdm[i,l]
                    new[j,l,i,k]+= -ij*d1.rdm[k,l]
                    new[j,l,i,k]+= kl*ij
    nRDM = RDM(
            order=self.p,
            alpha=self.alp,
            beta=self.bet,
            state='given',
            Ne=self.Ne,
            rdm=new,
            rdm_type='q',
            )
    return nRDM

def get_G_matrix(rdm):
    if not self.p==2:
        sys.exit('Not order 2 g matrix.')
    if not self.dqg=='d':
        sys.exit('Getting G from non-2RDM!.')
    self.expand()
    new = np.zeros(self.rdm.shape,dtype=np.complex_)
    d1 = self.reduce_order()
    for i in range(self.r):
        for j in range(self.r):
            for k in range(self.r):
                for l in range(self.r):
                    delta = (k==l)
                    new[i,l,j,k]+= -self.rdm[i,k,j,l]+delta*d1.rdm[i,j]
    nRDM = RDM(
            order=self.p,
            alpha=self.alp,
            beta=self.bet,
            state='given',
            Ne=self.Ne,
            rdm=new,
            rdm_type='g',
            )
    return nRDM
