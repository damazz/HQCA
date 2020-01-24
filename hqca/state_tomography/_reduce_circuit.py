import random
import sys

class PauliString:
    def __init__(self,term):
        self.p = term
        self.n = len(term)

    def __mul__(self,P):
        return self._qubit_wise_commuting(P)

    def _qubit_wise_commuting(self,P):
        if not self.n==P.n:
            return True
            #return False
        for i in range(self.n):
            if self.p[i]=='I' or P.p[i]=='I':
                pass
            elif self.p[i]==P.p[i]:
                pass
            else:
                #return False
                return True
        #return True
        return False

    def __str__(self):
        return self.p

class Graph:
    def __init__(self,
            vertices,
            edges,
            ):
        self.V = vertices
        self.edges = edges
        self._order_vertices()

    def _order_vertices(self):
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

    def _recursive_largest_first(self):
        self.coloring = {}
        done = False
        k = -1
        U = self.V[:]
        iteration = 0 
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
            
    def _rlf_generate_Cv(self,v,U):
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


def pauli_relation(A,B):
    test = PauliString(A)*PauliString(B)
    return test

def construct_simple_graph(items,related):
    N = len(items)
    edges = []
    for j in range(N):
        for i in range(j):
            if related(items[i],items[j]):
                edges.append([items[i],items[j]])
    return Graph(items,edges)


def simplify_tomography(operators,method='RLF',verbose=False):
    graph = construct_simple_graph(operators,pauli_relation)
    graph.color(method)
    new = {}
    for k,c in graph.coloring.items():
        try:
            new[c].append(k)
        except KeyError:
            new[c]=[k]
    paulis = {} #input a pauli, retrns the gate you need 
    ops = []
    for color, item in new.items():
        if len(item)==0:
            paulis[item[0]]=item[0]
        else:
            temp = item[0]
            for p in range(len(item)-1):
                temp = combine_strings(temp,item[p+1])
            for term in item:
                paulis[term]=temp
        ops.append(temp)
    if verbose:
        print('-- -- -- -- -- -- -- -- -- -- -- ')
        print('      --   TOMOGRAPHY   --      ')
        print('-- -- -- -- -- -- -- -- -- -- -- ')
        print('Distinct pauli terms: {}'.format(len(operators)))
        print('Distinct cliques: {}'.format(len(new)))
    return ops,paulis


