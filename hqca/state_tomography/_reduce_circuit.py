import random
import numpy as np
from functools import partial
import sys
import timeit
from copy import deepcopy as copy
import networkx as nx
import graph_tool as gt
from graph_tool import topology
from networkx.algorithms.coloring import *

class Graph:
    '''
    CLass for managing variety of graphing problems. Of note are two
    implementations for handling large graphs:
        1. networkx, and:
        2. graph_tool
    '''
    def __init__(self,
            generate=True,
            vertices=None,
            edges=None,
            verbose=False,
            graph=None,
            backend='nx',
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

        method should include 'greedy' (referring to networkx implementation)


        elsewhere 'gt' refers to graph_tools

        '''
        if method=='greedy':
            alg = greedy_color(self.g,strategy=strategy)
            self.colors = {}
            for k,v in alg.items():
                try:
                    self.colors[v].append(k)
                except Exception:
                    self.colors[v]=[k]
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
                for n,i in enumerate(alg.a):
                    try:
                        self.colors[i].append(n)
                    except Exception:
                        self.colors[i]=[n]

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

def qwc(A,B):
    for i in range(len(A)):
        if A[i]=='I' or B[i]=='I':
            pass
        elif A[i]==B[i]:
            pass
        else:
            return False
    return True


def commute(A,B):
    k=0
    for i in range(len(A)):
        if A[i]=='I' or B[i]=='I':
            pass
        elif A[i]==B[i]:
            pass
        else:
            k+=1
    return (k+1)%2

def pauli_relation(A,B,rel='qwc'):
    if rel=='qwc':
        return not qwc(A,B)
    elif rel=='mc':
        return not commute(A,B)
    else:
        return rel(A,B)

def construct_simple_graph(
        items,related,
        verbose=False,
        stochastic=False,
        backend='gt',
        **kw
        ):
    if backend=='gt':
        graph = gt.Graph(directed=False)
        N = len(items)
        j = 1
        n = 0
        edges = []
        while j<N-1:
            j+=1
            for i in range(j):
                if related(items[i],items[j]):
                    n+=1 
                    edges.append([i,j])
            if n//1e7>0 and n>0: #10 million edges 
                n-=1e7
                if sys.getsizeof(edges)>1e10:
                    # 10 gb of memory
                    print('Contracting')
                    print('Current size: {}'.format(sys.getsizeof(edges)))
                    graph.add_edge_list(edges)
                    edges = []
        graph.add_edge_list(edges)
        if verbose:
            print('Size of edge list (mem): {}'.format(sys.getsizeof(edges)))
        G = Graph(generate=False,graph=graph)
        if verbose:
            print('Size of graph (mem): {}'.format(sys.getsizeof(G)))
    elif backend=='nx':
        graph = nx.Graph()
        N = len(items)
        edges = []
        for j in range(N):
            for i in range(j):
                if related(items[i],items[j]):
                    graph.add_edge(i,j)
        G = Graph(generate=False,graph=graph)
        if verbose:
            print('Size of graph: {}'.format(sys.getsizeof(G)))
    G.rev_map = items
    return G

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
    if len(operators)==1:
        op = operators[0]
        return [op],{op:op}
    graph = construct_simple_graph(operators,relation,verbose=verbose,**kw)
    if verbose:
        try:
            print('Vertices: {}, Edges: {}'.format(
                graph.g.number_of_nodes(),
                graph.g.number_of_edges(),
                ))
        except Exception as e:
            print('Vertices: {}, Edges: {}'.format(
                graph.g.num_vertices(),
                graph.g.num_edges()
                )
                )
        t2 = timeit.default_timer()
        print('Time to make graph: {:.2f}'.format(t2-t1))
    graph.color(**kw)
    if verbose:
        t3 = timeit.default_timer()
        print('Time to color graph: {:.2f}'.format(t3-t2))
    c2p = []
    colors = {}
    sizes = []
    for k,v in graph.colors.items():
        V = [operators[i] for i in v]
        sizes.append(len(V))
        colors[k]=V
    for v in range(len(colors.keys())):
        c2p.append(__find_largest_qwc(colors[v]))
    ops= c2p[:]
    paulis = {}
    for k,v in graph.colors.items():
        K = ops[k]
        for p in v:
            paulis[graph.rev_map[p]]=K
    print('Number of colors: {}'.format(len(graph.colors.keys())))
    print('Largest coloring: {}'.format(max(sizes)))
    print('Standard deviation: {:.3f}'.format(np.std(np.asarray(sizes))))
    return ops,paulis

def compare_tomography(
        operators,
        verbose=False,
        rel='qwc',
        methods=['greedy'],
        strategies=['largest_first'],
        backend='gt',
        **kw):
    print('---  ---  ---  ---  ---  --- ')
    print('Comparison of different sorting algorithms: ')
    print('Relation: {}'.format(rel))
    print('Constructing graph...')
    t1 = timeit.default_timer()
    relation = partial(pauli_relation,**{'rel':rel})
    graph = construct_simple_graph(
            operators,
            relation,
            verbose=verbose,
            backend=backend,
            **kw)
    try:
        print('Vertices: {}, Edges: {}'.format(
            graph.g.number_of_nodes(),
            graph.g.number_of_edges(),
            ))
    except Exception:
        print('Vertices: {}, Edges: {}'.format(
            graph.g.num_vertices(),
            graph.g.num_edges()
            )
            )
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
                paulis[graph.rev_map[p]]=K
    return ops,paulis

