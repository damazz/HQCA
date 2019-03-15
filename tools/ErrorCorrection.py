

class Polyhedron:
    def __init(self,vertices):
        self.vert = np.asmatrix(vertices)
        self.d,self.p = vertices.shape
        if not self.d==self.p:
            print('Too many or too little points specified for the')
            print('desired affine transformation.')


    def affine(self,G):
        '''
        Generates an affine transformation from  A (self) to B (G)
        '''
        if not G.d==self.d:
            print('Error in transformations.')
            sys.exit()
        A = np.zeros((self.d+1,self.p))
        B = np.zeros((G.d+1,G.p))
        A[:self.d+1:]=self.vert[:,:]
        A[self.d+1,:]=np.ones(self.p)
        B[:G.d+1:]=G.vert[:,:]
        B[G.d+1,:]=np.ones(G.p)
        Ai = np.linalg.inv(A)
        return np.dot(B,Ai)



class CompositePolytope
    '''

    Somewhat of a generalization of the triangles/planes to larger surfaces.
    Object which has multiple planes, 

    Error correction via projection or correction onto accessible hyperplane.
    Currently configured for only certain types of problems. 

    Procedure is as follows. For a list of parameters, generate the measurable
    points. Then, we generate the hyperplane object, add these measurable
    triangles to is, and then 
    '''
    def __init__()
        self.Nface = 0 
        self.poly = {}


    def map(self,point):
        self._map_to_nearest_face(point)

    def _map_to_nearest_face(self,point):
        target = 0 
        min_dist =  1 


    def add_face(self,ideal_face,measured_face):
        pass





def generate_hyperplane():
