import random
import numpy as np
from functools import partial
import sys
import timeit
from copy import deepcopy as copy
from hqca.tomography.__graph import *


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
        graph.add_vertex(N)
        j = 0
        n = 0
        edges = []
        single = []
        # constructing the edges of the graph 
        while j<N-1:
            j+=1
            rel = False
            for i in range(j):
                if related(items[i],items[j]):
                    n+=1
                    edges.append([i,j])
                    rel=True
            if n//1e7>0 and n>0:
                # here, we check if the list is getting too large, then we will add it to the graph, 
                # which stores it in C++, and continue with a new edge list 
                # 10,000,000 edges is current list 
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
        #G.singular = single
        if verbose:
            print('Size of graph (mem): {}'.format(sys.getsizeof(G)))
    G.rev_map = items
    return G

def __find_largest_qwc(A):
    '''
    given a set A, returns the macro QWC pair
    '''
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
    # set relation 
    relation = partial(pauli_relation,**{'rel':rel})
    # if we only have a single operator, return the graph 
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
    #
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
    if verbose:
        print('Number of colors: {}'.format(len(graph.colors.keys())))
        print('Largest coloring: {}'.format(max(sizes)))
        print('Standard deviation: {:.3f}'.format(np.std(np.asarray(sizes))))
    # 
    # check
    error = False
    for p in paulis.keys():
        meas = paulis[p]
        match = True
        for pi,mi in zip(p,meas):
            if pi=='I':
                continue
            else:
                if not mi==pi:
                    match=False
        if not match:
            print('Measurement does not match.')
            print('M: ',meas,'P:',p)
            error = True
    if error:
        sys.exit('Error in generating graph.')
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

