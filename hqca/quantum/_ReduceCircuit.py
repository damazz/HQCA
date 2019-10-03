from hqca.tools.Fermi import FermiOperator as Fermi
import random
import sys

def _qubit_wise_commuting(pauliA,pauliB):
    for i in range(len(pauliA)):
        if pauliB[i] not in _qwc_terms(pauliA[i]):
            return False
        else:
            pass
    return True

def _qwc_terms(pA):
    if pA in ['I']:
        return ['I','Z','X','Y']
    elif pA in ['Z']:
        return ['I','Z']
    elif pA in ['Y']:
        return ['I','Y']
    elif pA in ['X']:
        return ['I','X']

def combine_strings(pA,pB):
    temp = []
    for i in range(len(pA)):
        c1,c2 = pA[i]=='I', pB[i]=='I'
        if c1 and c2:
            temp.append('I')
        elif c1:
            temp.append(pB[i])
        elif c2:
            temp.append(pA[i])
        else:
            temp.append(pA[i])
    return ''.join(temp)


def simplify_tomography(rdm_elements,simplify='default',use_random=False):
    N = len(rdm_elements[0].pauliGet[0])
    temp_list = []
    done = False
    compare = []
    for fermi in rdm_elements:
        for j in fermi.pauliGates:
            temp_list.append(j)
    while not done:
        restart=False
        done = True
        for n in range(len(temp_list)):
            for m in range(n):  #m < n
                qwc  = _qubit_wise_commuting(
                        temp_list[n],
                        temp_list[m])
                if qwc:
                    new = combine_strings(
                            temp_list[n],
                            temp_list[m])
                    c1, c2 = new==temp_list[n], new==temp_list[m]
                    if c1 and c2:
                        temp_list.pop(n)
                    elif c1:
                        temp_list.pop(m)
                    elif c2:
                        temp_list.pop(n)
                    else:
                        temp_list.pop(n)
                        temp_list.pop(m)
                        temp_list.append(new)
                    restart = True
                    done=False
                    break
            if restart:
                break
    #print(temp_list)
    #print(compare)
    #print(len(temp_list),len(compare))
    #sys.exit()
    return rdm_elements


def find_optimal_mapping(alpha,beta):
        rdme = []
        S = []
        blocks = [
                [alpha,alpha,beta],
                [alpha,beta,beta],
                [alpha,beta,beta],
                [alpha,alpha,beta]
                ]
        block = ['aa','ab','bb']
        for ze in range(len(blocks[0])):
            for i in blocks[0][ze]:
                for k in blocks[1][ze]:
                    for l in blocks[2][ze]:
                        for j in blocks[3][ze]:
                            if block[ze]=='ab':
                                if i>j or k>l:
                                    continue
                                spin = ['abba']
                            else:
                                if i>=k or j>=l:
                                    continue
                                if block[ze]=='aa':
                                    spin = ['aaaa']
                                else:
                                    spin = ['bbbb']
                            test = Fermi(
                                coeff=1,
                                indices=[i,k,l,j],
                                sqOp='++--',
                                spin=spin[0])
                            test.generateTomoBasis(
                                    real=real,
                                    imag=imag,
                                    Nq=self.qs.Nq)
                            rdme.append(test)
pass
