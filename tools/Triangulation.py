import subprocess
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import time
import timeit
import sys
np.set_printoptions(precision=6,suppress=True)
from hqca.tools.QuantumFramework import build_circuits,run_circuits,Construct
from hqca.tools.QuantumFramework import wait_for_machine

#
# Triangulation functions
#

def get_points(
        f,
        Nt,
        start,
        stop,
        pr_t=0,
        thresh=1e-2):
    '''
    Start and stop are parameters
    '''
    length = 1
    diff = 1
    lines = 1
    target = np.linspace(0,1,Nt+1)
    points = target[1:-1]
    Np = len(points)
    #print(points)
    z = 0
    conv = False
    while (not conv):
        try: 
            oldf = finalf
            oldx = finalx
        except UnboundLocalError:
            pass
        finalf =  []
        finalx =  []
        lines*=2
        temp = 0
        pi = start+(stop-start)/lines
        pf = stop- (stop-start)/lines
        starts = np.linspace(pi,stop,lines+1)
        ends = np.linspace(start,pf,lines+1)
        point_ind = 0 
        for i in range(0,lines):
            dL = np.sqrt(
                    np.sum(
                        np.square(
                            f(starts[i])-f(ends[i])
                            )
                        )
                    )
            try:
                check1 = (temp/length)<=points[point_ind]
                check2 = points[point_ind]<=((temp+dL)/length)
                if (check1 and check2):
                    finalf.append(f(starts[i]))
                    finalx.append(starts[i])
                    point_ind +=1 
            except IndexError:
                pass
            temp += dL
        diff = length-temp
        if pr_t>1:
            print('Iteration: {}, Length: {}, Difference:{}'.format(
                z,temp,abs(diff
                        )
                    )
                )
            try:
                if pr_t>0:
                    print('Difference in points: {}'.format(abs(oldx[0]-finalx[0])))
            except UnboundLocalError:
                if pr_t>0:
                    print('--- Final f,x: {},{} --- '.format(finalf,finalx))
        z+=1 
        length = temp
        try:
            if (abs(diff)<thresh and abs((oldx[0]-finalx[0]))<thresh):
                conv=True
        except UnboundLocalError:
            pass
    finalx = np.asarray(finalx).tolist()
    return finalx



def project_to_plane(v1,v2,v3,point):
    '''
    Projects a point onto a plane specified by three points, and also calculates
    the distance to the plane. 
    '''
    u = v1-v2
    v = v2-v3
    plane_og = np.cross(u,v) 
    plane_on = plane_og/(np.sqrt(np.sum(np.square(plane_og))))
    w_vec = point-v1
    c = np.dot(plane_on,w_vec.T)
    mag = np.sqrt(np.sum(np.square(c*plane_on)))
    proj = point - c*plane_on
    return proj,mag

class CompositeAffine:
    def __init__(self):
        '''
        Class for constructing triangulations of surfaces. 
        '''
        self.triangles = {}
        self.Ntri = 0

    def map(self,point):
        '''
        Using the existing maps, find the plane describing the closest
        transformation to the point, and then returns the unitary affine matrix
        to perform the transformation.
        '''
        target = 0 
        min_dist = 1
        s = 'Distances: '
        for i in range(0,self.Ntri):
            T = self.triangles['tri{}'.format(i)]['input']
            T2 = self.triangles['tri{}'.format(i)]['output']
            proj, cur_dist = project_to_plane(T.v1,T.v2,T.v3,point)
            if cur_dist<min_dist:
                target=i
                min_dist = cur_dist
            s += '{},{:.4f}; '.format(i,cur_dist)
        return self.triangles['tri{}'.format(target)]['aU']

    def add_triangle(self,x,fx):
        '''
        Takes points describing two triangles, and constructs mapping for them. 
        '''
        tri1 = Triangle(x[0],x[1],x[2])
        tri2 = Triangle(fx[0],fx[1],fx[2])
        aU = tri1.affine(tri2)
        self.triangles['tri{}'.format(self.Ntri)]={
                'input':tri1,
                'output':tri2,
                'aU':aU
                }
        self.Ntri += 1



#
#
#
#

class Triangle:
    def __init__(self,v1,v2,v3):
        ''' 
        Triangle class.
        Each object formed from three vectors. To get an affine from two
        triangles, where U:A->B, then we calculate A.affine(B). 

        '''
        self.v1 = np.asmatrix(v1)
        self.v2 = np.asmatrix(v2)
        self.v3 = np.asmatrix(v3)
        self.m  = ((1/3)*(self.v1+self.v2+self.v3)).T

    def affine(self,T):
        A = np.zeros((4,3))
        B = np.zeros((4,3))
        A[:3,0]=self.v1
        A[:3,1]=self.v2
        A[:3,2]=self.v3
        A[3,:]=np.matrix([[1,1,1]])
        B[:3,0]=T.v1
        B[:3,1]=T.v2
        B[:3,2]=T.v3
        B[3,:]=np.matrix([[1,1,1]])
        Ai = np.linalg.pinv(A)
        return np.dot(B,Ai)

    def transform(self,T,point):
        U = self.affine(T)
        try:
            point = np.asmatrix(point)
            x = U*point
        except Exception:
            point  = np.asmatrix(point).T
            x = U*point
        return x[0:3,0]

def find_triangle(
        qc_backend,
        wf_mapping,
        store,
        method,
        algorithm,
        Ntri='default',
        pr_t=0,
        wait_for_runs=True,
        energy='qc',
        **kwargs
        ):
    '''
    Procedure for measuring energy points in the simplex spanned by the
    parameters of the generating algorithm i.e. 'triangle' if using a 2
    paramaeter algorithm. For use in an affine transformation approach to the
    mappping onto the GPC plane. 

    Currently, configured just for the algorithm 'ry2p_raven_diag'.  
    '''
    kwargs['algorithm']=algorithm
    if algorithm in ['affine_2p_flat_tenerife','affine_2p_flatfish_tenerife']:

        if algorithm=='affine_2p_flatfish_tenerife':
            par = [[0,0],[0,45],[45,0]]
        else:
            par = [[0,0],[0,45],[45,45]]
        tri = []
        kwargs['backend']=backend
        for item in par:
            if wait_for_runs and (backend in ['ibmqx4','ibmqx2']):
                wait_for_machine(backend)
            else:
                pass
            kwargs['para']=item
            qc,qcl = build_circuits(para,**kwargs)
            qco = run_circuits(qc,qcl,**kwargs)
            rdm1 = construct(qco,**kwargs)
            on = data.on
            rdm = data.rdm1
            on.sort()
            tri.append([on[5],on[4],on[3]])
        affine = CompositeAffine()
        affine.add_triangle(
                [tri[0],tri[1],tri[2]],
                [   [1,1,1],
                    [1,0.5,0.5],
                    [0.75,0.75,0.5]]
                )
        if pr_t>0:
            print('Triangle: \n{}\n{}\n{}'.format(t1.v1,t1.v2,t1.v3))
            print('Transformation: \n {}'.format(aU))
    elif algorithm=='affine_2p_curved_tenerife':
        def gpc(t):
            x0 = np.array([1,1,1])
            x1 = np.array([0.75,0.75,0.5])
            return x0 + t*(x1-x0)

        def f(the):
            the = the*np.pi/180
            x = np.cos(the)**2
            y = (1-x)**2 + x**2
            return np.array([x,x,y])

        if Ntri == 'default':
            Ntri = 2
        par_list = [[0,45],[0,0]]
        targets = [[1,1,1]]
        if Ntri>1:
            tri_par = get_points(f,Ntri,0,45,pr_t=pr_t)
            tar_tri = get_points(gpc,Ntri,0,1,pr_t=pr_t)
            for i in tri_par: 
                try:
                    par_list.append([i[0],i[0]])
                except TypeError:
                    par_list.append([i,i])
            for i in tar_tri:
                targets.append(gpc(i).tolist())
        par_list.append([45,45])
        targets.append([0.75,0.75,0.5])
        affine = CompositeAffine()
        if pr_t>1:
            print('Targets: {}'.format(targets))
        for i in range(0,Ntri):
            par = [par_list[0]]+par_list[i+1:i+3]
            if pr_t>1:
                print('Parameters: {}'.format(par))
            tri = []
            kwargs['qc_backend']=qc_backend
            for item in par:
                if wait_for_runs and (qc_backend in ['ibmqx4','ibmqx2']):
                    from qiskit import register
                    try:
                        import Qconfig
                    except ImportError:
                        from ibmqx import Qconfig
                    try:
                        register(Qconfig.APItoken)
                    except Exception as e:
                        print(e)
                        pass
                    wait_for_machine(backend)
                else:
                    pass
                kwargs['para']=item
                qc,qcl,q = build_circuits(**kwargs)
                kwargs['qb2so']=q
                qco = run_circuits(qc,qcl,**kwargs)
                rdm = construct(qco,**kwargs)
                on,norbs = np.linalg.eig(rdm)
                on.sort()
                if pr_t>1:
                    print('Vertex: {}'.format(on))
                tri.append([on[5],on[4],on[3]])
            affine.add_triangle(
                    [tri[0],tri[1],tri[2]],
                    [   [1,0.5,0.5],
                        targets[i],
                        targets[i+1]]
                    )
    return affine


