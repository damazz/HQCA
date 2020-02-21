import random
import numpy as np
from functools import partial
import sys
import timeit
from copy import deepcopy as copy
import networkx as nx

class PauliString:
    def __init__(self,term):
        self.p = term
        self.n = len(term)

    def __mul__(self,P):
        return self._qubit_wise_commuting(P)

    def _qubit_wise_commuting(self,P):
        if not self.n==P.n:
            return False
        for i in range(self.n):
            if self.p[i]=='I' or P.p[i]=='I':
                pass
            elif self.p[i]==P.p[i]:
                pass
            else:
                #return False
                return False
        #return True
        return True

    def __str__(self):
        return self.p

    def _commuting(self,P):
        k=0
        if not self.n==P.n:
            return False
        for i in range(self.n):
            if self.p[i]=='I' or P.p[i]=='I':
                pass
            elif self.p[i]==P.p[i]:
                pass
            else:
                k+=1
        return (k+1)%2


from networkx.algorithms.coloring import * 

class Graph:
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
            method='greedy',
            strategy='largest_first',
            **kwargs):
        if method=='greedy':
            alg = greedy_color(self.g,strategy=strategy)
        self.colors = {}
        for k,v in alg.items():
            try:
                self.colors[v].append(k)
            except Exception:
                self.colors[v]=[k]
    
class oldGraph:
    def __init__(self,
            vertices,
            edges,
            verbose=True,
            **kw
            ):
        '''
        self.V is a list of veritices
        self.edge is a list of lists, where each entry has the vertices which
        compose the edge
        '''
        self.V = vertices
        self.edges = edges
        print(self.edges)
        self._order_vertices()
        self.Nv = len(self.V)
        self.Ne = len(self.edges)
        self.verbose=verbose
        


    def _order_vertices(self):
        '''
        orders the list of vertices and finds the degree, indicating how many
        edges it is included in, and paths indicates the possible connections
        that each vertices can link to.
        '''
        self.degree = {k:0 for k in self.V}
        self.paths = {k:[] for k in self.V}
        for i,j in self.edges:
            self.degree[i]+=1
            self.degree[j]+=1
            self.paths[i].append(j)
            self.paths[j].append(i)
        self.degree_sorted = sorted(
                self.degree.items(),key=lambda item:item[1])

    def color(self,method='RLF'):
        if method=='RLF':
            self._recursive_largest_first()
        elif method=='fRLF':
            self._faster_rlf()

    def _faster_rlf(self):
        pass
    
    def _recursive_largest_first(self):
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
        U = self.V[:]
        iteration = 0
        if self.verbose:
            t0 = timeit.default_timer()
        while len(self.coloring)<len(self.V):
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

def combine_strings(A,B):
    def _delta(a,b):
        if a=='I':
            return b
        elif b=='I':
            return a
        else:
            return a
    hold = []
    for i in range(len(A)):
        hold.append(_delta(A[i],B[i]))
    return ''.join(hold)

def pauli_relation(A,B,rel='qwc'):
    if rel=='qwc':
        test = not PauliString(A)*PauliString(B)
    elif rel=='mc':
        test = not PauliString(A)._commuting(PauliString(B))
    return test

def construct_simple_graph(
        items,related,
        verbose=False,
        stochastic=False,
        threshold=0.1,
        **kw
        ):
    graph = nx.Graph()
    if stochastic:
        N = len(items)
        edges = np.ones((N,N))
        #edges = [[i,j for j in xrange(N)] for i in xrange(i)]
        k = 0
        for j in range(N):
            i = 0
            while i<threshold*N:
                rand = random.randint(0,N-1)
                if not related(items[j],items[rand]):
                    edges[j,rand]=0
                    edges[rand,j]=0
                    k+=1
                i+=1
        edges = np.nonzero(np.tril(edges))
        for i,j in zip(edges[0],edges[1]):
            graph.add_edge(i,j)
        print('{} edges removed '.format(k))
    else:
        N = len(items)
        edges = []
        for j in range(N):
            for i in range(j):
                if related(items[i],items[j]):
                    graph.add_edge(i,j)
    return Graph(generate=False,graph=graph)


def __find_largest_qwc(A):
    string = 'I'*len(A[0])
    for j in range(len(string)):
        done=False
        while not done:
            for i in A:
                if not i[j]=='I':
                    string = string[:j]+i[j]+string[j+1:]
                    done=True
                    break
            done=True
    return string

def simplify_tomography(
        operators,
        verbose=False,
        rel='qwc',
        **kw):
    if verbose:
        print('Relation: {}'.format(rel))
        print('Constructing graph...')
        t1 = timeit.default_timer()
    relation = partial(pauli_relation,**{'rel':rel})
    graph = construct_simple_graph(operators,relation,verbose=verbose,**kw)
    if verbose:
        print('Vertices: {}, Edges: {}'.format(
            graph.g.number_of_nodes(),
            graph.g.number_of_edges(),
            ))
        t2 = timeit.default_timer()
        print('Time to make graph: {:.2f}'.format(t2-t1))
    graph.color(**kw)
    if verbose:
        t3 = timeit.default_timer()
        print('Time to color graph: {:.2f}'.format(t3-t2))
    c2p = []
    colors = {}
    for k,v in graph.colors.items():
        V = [operators[i] for i in v]
        colors[k]=V
    for v in range(len(colors.keys())):
        c2p.append(__find_largest_qwc(colors[v]))
    ops= c2p[:]
    paulis = {}
    for k,v in graph.colors.items():
        K = ops[k]
        for p in v:
            paulis[p]=K
    return ops,paulis

def compare_tomography(
        operators,
        verbose=False,
        rel='qwc',
        methods=['greedy'],
        strategies=['largest_first'],
        **kw):
    print('---  ---  ---  ---  ---  --- ')
    print('Comparison of different sorting algorithms: ')
    print('Relation: {}'.format(rel))
    print('Constructing graph...')
    t1 = timeit.default_timer()
    relation = partial(pauli_relation,**{'rel':rel})
    graph = construct_simple_graph(operators,relation,verbose=verbose,**kw)
    if verbose:
        print('Vertices: {}, Edges: {}'.format(
            graph.g.number_of_nodes(),
            graph.g.number_of_edges(),
            ))
        t2 = timeit.default_timer()
        print('Time to make graph: {:.3f}'.format(t2-t1))
    for method,strategy in zip(methods,strategies):
        print('Comparison on colorings and clique sizes')
        t2 = timeit.default_timer()
        graph.color(method=method,
                strategy=strategy)
        if verbose:
            t3 = timeit.default_timer()
            print('Method: {}, Strategy: {}, Time: {:.3f}'.format(
                method,strategy,
                t3-t2))
        c2p = []
        colors = {}
        sizes = []
        for k,v in graph.colors.items():
            V = [operators[i] for i in v]
            sizes.append(len(V))
            colors[k]=V
        print('Number of colors: {}'.format(len(graph.colors.keys())))
        print('Largest coloring: {}'.format(max(sizes)))
        print('Standard deviation: {:.3f}'.format(np.std(np.asarray(sizes))))
        for v in range(len(colors.keys())):
            c2p.append(__find_largest_qwc(colors[v]))
        ops= c2p[:]
        paulis = {}
        for k,v in graph.colors.items():
            K = ops[k]
            for p in v:
                paulis[p]=K
    return ops,paulis

