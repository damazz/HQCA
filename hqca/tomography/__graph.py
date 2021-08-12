import networkx as nx
import graph_tool as gt
from graph_tool import topology

class Graph:
    '''
    CLass for managing variety of graphing problems. We use networkx and graph_tool,
    the latter of which is the default method, and includes a native coloring
    algorithm.

    '''
    def __init__(self,
            generate=True,
            vertices=None,
            edges=None,
            verbose=False,
            graph=None,
            **kwargs):
        if generate:
            self.g = nx.Graph()
            for (v,e) in edges:
                self.g.add_edge(v,e)
        else:
            self.g = graph

    def color(self,
            method='gt',
            strategy='largest_first',
            **kwargs
            ):
        '''
        selects coloring method and strategy from various options

        default is graphtools implementation

        '''
        if method=='greedy':
            alg = greedy_color(self.g,strategy=strategy)
            self.colors = {}
            nc = 0
            for k,v in alg.items():
                try:
                    self.colors[v].append(k)
                except Exception as e:
                    self.colors[v]=[k]
                    print(e)
                nc+=1 
        elif method=='gt':
            if strategy in ['default']:
                self.alg = gt.topology.sequential_vertex_coloring(self.g)
                for n,i in enumerate(self.alg.a):
                    try:
                        self.colors[i].append(n)
                    except Exception:
                        self.colors[i]=[n]
            elif strategy in ['rlf']:
                self.recursive_largest_first()
            elif strategy in ['largest_first','lf']:
                v = self.g.get_vertices()
                g = self.g.get_total_degrees(self.g.get_vertices())
                ordering = sorted(v,key=lambda i:g[i],reverse=True)
                ord_vpm = self.g.new_vertex_property('int')
                for n,i in enumerate(ordering):
                    ord_vpm[n]=i
                alg = gt.topology.sequential_vertex_coloring(
                        self.g,
                        order=ord_vpm,
                        )
                self.colors = {}
                nc = 0
                for n,i in enumerate(alg.a):
                    try:
                        self.colors[i].append(n)
                    except Exception: 
                        self.colors[i]=[n]
                        nc+=1

    def recursive_largest_first(self):
        pass
        '''
        based on "A new efficient RLF-like algorithm for the Vertex Coloring
        Problem." Adegbindin, Hertz, Bellaiche. 2015
        
        note, we need to convert the graph to its complement graph, which then
        can be mapped to a coloring problem

        C refers to a color class under construction, with two sets: W and V,
        which represent uncolored vertices and uncolored vertices with neighbors
        in C. repectively. First v in U has largest number of neighbors in U.
        Then, while U is not empty, find w in U with largest neighbors in W.
        Then move that to C and also move neigbhors of w to W. When U is empty,
        proceed to next color class.
        '''
        self.coloring = {}
        done = False
        k = -1
        N = self.g.num_vertices()
        N_assigned = 0
        assigned = self.g.new_vertex_property('bool')
        colors = self.g.new_vertex_property('int')
        while N_assigned<N:
            self.g.set_vertex_filter(assigned,inverted=True)
            k+=1
            vertices = self.g.get_vertices()
            degrees = self.g.get_total_degrees(vertices)
            v2i = {i:n for n,i in enumerate(vertices)}
            lf = sorted(
                    vertices,
                    key=lambda i:degrees[v2i[i]],
                    reverse=True)
            v = lf[0]
            condition=True
            # Cv
            Nu = self.g.num_vertices()
            W = self.g.new_vertex_property('bool')
            Aw = self.g.new_vertex_property('int')
            Au = self.g.get_total_degrees(vertices)
            W[v]=1
            for i in self.g.get_all_neighbors(v):
                W[i]=1
                Aw[i]+=1
                Au[v2i[i]]-=1
            assigned[v]=1
            N_assigned+=1
            colors[v]=k 
            j=0
            while np.sum(W.get_array())<Nu and j<10:
                j+=1
                # sort according to W
                # element in U: W=0 
                def sort_U(i):
                    a = (1-W.get_array()[i])
                    b = Aw[i]
                    return (a,b)
                large_w = sorted(
                        vertices,
                        key=lambda i:sort_U(i),
                        reverse=True)
                u = large_w[0]
                assigned[u]=1
                N_assigned+=1
                colors[u]=k
                neighbor_u  = self.g.get_all_neighbors(u)
                Aw[u]=1
                W[u]=1
                for i in neighbor_u:
                    W[i]=1
                    Aw[i]+=1
        self.colors = {}
        for n,c in enumerate(colors.get_array()):
            try:
                self.colors[c].append(n)
            except Exception:
                self.colors[c]=[n]
        self.g.clear_filters()

    def _find_uncolored_vertices():
        pass
        #iteration = 0
        if self.verbose:
            t0 = timeit.default_timer()
        while len(self.coloring.keys())<len(N):
            iteration+=1
            # get Au
            k+=1
            Au = {}
            for u in U:
                Au[u]=0
                for uu in U:
                    if uu in self.paths[u] and not u==uu:
                        Au[u]+=1
            v = sorted(Au.items(),key=lambda k:k[1])[-1][0]
            Cv,U = self._rlf_generate_Cv(v,U)
            for i in Cv:
                self.coloring[i]=k
            if iteration%10==0:
                if self.verbose:
                    t = timeit.default_timer()
                    print('Time after 10 iterations: {}'.format(t-t0))
                    t0 = copy(t)

    def _rlf_generate_Cv(self,v,U):
        '''
        subroutine to geneate Cv:
        '''
        W = []
        Cv = [v]
        U.remove(v)
        for u in reversed(U):
            if v in self.paths[u]:
                W.append(u)
                U.remove(u)
        while len(U)>0:
            Aw = {}
            for u in U:
                Aw[u]=0
                neighbor = self.paths[u]
                for uu in W:
                    if uu in neighbor and not uu==u:
                        Aw[u]+=1
            u = sorted(Aw.items(),key=lambda k:k[1])[-1][0]
            Cv.append(u)
            U.remove(u)
            for ur in reversed(U):
                if u in self.paths[ur]:
                    W.append(ur)
                    U.remove(ur)
        return Cv,W[:]
