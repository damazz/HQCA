'''
quantum/ErrorCorreciton.py

Used to implement the hyperplane error correction, i.e. projection onto a GPC
plane. Does not include other forms of error correction, although certain types
of post correction might be suitable. 

'''

import numpy as np
import sys
from hqca.quantum.QuantumFramework import build_circuits,run_circuits,Construct

class HyperPlane:
    '''
    Specify input vertices as column vectors 
    '''
    def __init__(self,vertices):
        self.vert = np.asmatrix(vertices)
        self.d,self.p = vertices.shape
        self.diff = np.asmatrix(np.zeros((self.d,self.p-1)))
        for i in range(0,self.p-1):
            self.diff[:,i]=self.vert[:,i]-self.vert[:,self.p-1]
        if not self.d==self.p:
            print('Too many or too little points specified for the')
            print('desired affine transformation.')
        self._find_normal_vector()

    def affine(self,G):
        '''
        Generates an affine transformation from  A (self) to B (G)
        '''
        if not G.d==self.d:
            print('Error in transformations.')
            sys.exit()
        A = np.zeros((self.d+1,self.p))
        B = np.zeros((G.d+1,G.p))
        A[:self.d,:]=self.vert[:,:]
        A[self.d,:]=np.ones(self.p)
        B[:G.d,:]=G.vert[:,:]
        B[G.d,:]=np.ones(G.p)
        Ai = np.linalg.pinv(A)
        return np.dot(B,Ai)

    def _find_normal_vector(self):
        self.n = np.zeros((self.d,1))
        if self.d==self.p:
            for i in range(0,self.d):
                temp = np.zeros((self.d-1,self.p-1))
                temp[:i,:]=self.diff[:i,:]
                temp[i:,:]=self.diff[i+1:,:]
                self.n[i] = ((-1)**i)*np.linalg.det(temp)
        else:
            sys.exit('Error')
        c = np.sqrt(np.sum(np.square(self.n)))
        self.n = (1/c)*self.n
        self._norm_check()

    def _norm_check(self):
        for i in range(0,self.p-1):
            temp = self.diff[:,i]
            if abs(np.dot(self.n.T,temp))>1e-10:
                sys.exit('Error in finding a normal vector. Check vectors.')

    def dist(self,v):
        '''
        Shortcut to function: 'distance_to_plane'
        '''
        return self.distance_to_plane(v)

    def distance_to_plane(self,v):
        '''
        Point on plane - p
        Plane vector 
        '''
        p = self.vert[:,0]
        n = np.dot(self.n.T,v-p)[0,0]*self.n
        return np.sqrt(np.sum(np.square(n)))

class CompositePolytope:
    '''
    Somewhat of a generalization of the triangles/planes to larger surfaces.
    Object which has multiple planes, 

    Error correction via projection or correction onto accessible hyperplane.
    Currently configured for only certain types of problems. 

    Procedure is as follows. For a list of parameters, generate the measurable
    points. Then, we generate the hyperplane object, add these measurable
    triangles to is, and then 
    '''
    def __init__(self,
            dim=4
            ):
        self.Nf = 0 #Num of faces
        self.poly = []
        self.d = dim

    def map(self,point):
        return self._map_to_nearest_face(point)

    def _map_to_nearest_face(self,point):
        N = 0
        min_dist =  1
        for i,f in enumerate(self.poly):
            dist = f['meas'].dist(point)
            if dist<min_dist:
                N=i
                min_dist = dist
        p = np.zeros((self.d+1,1))
        p[:self.d,0] = np.asmatrix(point)
        p[-1,0] = 1
        return np.dot(self.poly[N]['Umi'],p)

    def add_face(self,ideal_face,measured_face):
        '''
        Add a HyperPlane class object, and populate a polytope object
        Affine transformation is from measured to ideal
        '''
        self.poly.append({})
        self.poly[self.Nf]['ideal']=ideal_face
        self.poly[self.Nf]['meas']=ideal_face
        self.poly[self.Nf]['Umi'] = measured_face.affine(ideal_face)
        self.Nf+=1

def generate_error_polytope(QuantStore):
    '''
    Function to generate polytope with hyperplanes.
    '''
    ec_a = CompositePolytope(dim=len(QuantStore.alpha_qb))
    ec_b = CompositePolytope(dim=len(QuantStore.beta_qb))
    if QuantStore.pr_e>0:
        print('Obtaining error correction polytope.')
    for i in range(QuantStore.ec_Ns): #number of surface? 
        tempa = np.zeros((
            len(QuantStore.alpha_qb),
            len(QuantStore.alpha_qb))
            )
        tempb = np.zeros((
            len(QuantStore.beta_qb),
            len(QuantStore.beta_qb))
            )
        for j in range(QuantStore.ec_Nv): #number of vertices
            if QuantStore.hyperplane_custom:
                print('Using user provided polytope.')
                tempa[:,j]=QuantStore.hyper_alp[:,j]
                tempb[:,j]=QuantStore.hyper_bet[:,j]
            else:
                QuantStore.parameters = np.asarray(
                        QuantStore.ec_para[i][j])
                try:
                    QuantStore.parameters = QuantStore.error_shift[j,:]
                    #print('Shifted by: ')
                    print(QuantStore.error_shift[j,:])
                except Exception as e:
                    #print(e)
                    pass
                q_circ,qc_list = build_circuits(QuantStore)
                qc_obj = run_circuits(
                        q_circ,
                        qc_list,
                        QuantStore)
                proc = Construct(
                        qc_obj,
                        QuantStore)
                rdm1 = proc.rdm1
                if QuantStore.spin_mapping in ['default','alternating']:
                    No = rdm1.shape[0]
                    rdma = rdm1[:No//2,:No//2]
                    rdmb = rdm1[No//2:,No//2:]
                    noca,nora = np.linalg.eig(rdma)
                    nocb,norb = np.linalg.eig(rdmb)
                    N = len(noca)
                    idxa = np.argsort(noca)[::-1]
                    idxb = np.argsort(nocb)[::-1]
                    noca = noca[idxa]
                    nocb = nocb[idxb]
                    tempa[:,j]=noca
                    tempb[:,j]=nocb
                    if QuantStore.pr_e>1:
                        print('Parameters: ')
                        print(QuantStore.ec_para[i][j])
                        print('Experimental occupations: ')
                        print('Alpha: ')
                        print(noca)
                        print('Beta: ')
                        print(nocb)
        alp_hp = HyperPlane(tempa)
        bet_hp = HyperPlane(tempb)
        ideal = HyperPlane(QuantStore.ec_vert)
        ec_a.add_face(ideal,alp_hp)
        ec_b.add_face(ideal,bet_hp)
        if QuantStore.pr_e>1:
            print('Alpha hyperplane:')
            print(tempa)
            print('Beta hyperplane:')
            print(tempb)
            print('Ideal hyperplane: ')
            print(QuantStore.ec_vert)
        QuantStore._gip()
    QuantStore.ec_a = ec_a
    QuantStore.ec_b = ec_b

