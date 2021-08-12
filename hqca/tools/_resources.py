from hqca.operators import *

####
# qACSE related tools 
####

def count_cnot_resources(ansatz):
    upper = 0
    lower = 0
    if isinstance(ansatz,type(Operator())):
        for pauli in ansatz:
            l = 0
            for s in pauli.s:
                if not s == 'I':
                    l += 1
            if l==0:
                continue
            upper += 2 * (l - 1)
    else:
        for item in ansatz:
            if isinstance(item,type(Operator())):
                for pauli in item:
                    l = 0
                    for s in pauli.s:

                        if not s=='I':
                            l+=1
                    upper+= 2*(l-1)
    return upper, lower

