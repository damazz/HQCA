import pickle
import numpy as np
import qiskit
import sys
###
# data keys:
# shots,backend,maxCredits,deleted,qasms,userId,calibrations,status,creationDate,usedCredits,id
###
def earlier(time_one,time_two):
    #2018-03-04T04:23:51.103Z
    # basic format
    # is  one earlier than two?
    t1 = time_one.split('T')
    t2 = time_two.split('T')
    t1_a = t1[0].split('-')
    t1_b = t1[1][:-1].split(':')
    t2_a = t2[0].split('-')
    t2_b = t2[1][:-1].split(':')
    one = []
    two = []
    for i in range(0,3):
        one.append(float(t1_a[i]))
        two.append(float(t2_a[i]))
    for i in range(0,3):
        one.append(float(t1_b[i]))
        two.append(float(t2_b[i]))
    #printone,two)

    if one[0]<two[0]:
        return True
    elif one[0]==two[0]:
        if one[1]<two[1]:
            return True
        elif one[1]==two[1]:
            if one[2]<two[2]:
                return True
            elif one[2]==two[2]:
                if one[3]<two[3]:
                    return True
                elif one[3]==two[3]:
                    if one[4]<two[4]:
                        return True
                    elif one[4]==two[4]:
                        if one[5]<two[5]:
                            return True
                        else:
                            return False
                    else:        
                        return False
                else:
                    return False
            else:        
                return False
        else:        
            return False
    else:
        return False
   

test = ['22','49','07']

with open(sys.argv[1],'rb') as fp:
    data = pickle.load(fp)
i = 0
n = len(data)

# these will be the data storage sets

sort_1 = []
sort_2 = []
sort_3  = []

# first, sorting by time so that the data is properly sorted
sort_1.append(data[0])
print(sort_1[0].keys())
for i in range(1,n):
    time_one = data[i]['creationDate']
    ind  =0 
    for j in sort_1:
        time_two = j['creationDate']
        # assume new time is later than first one
        if earlier(time_one,time_two):
            try:
                sort_1 = sort_1[:ind] + [data[i]] + sort_1[ind:]
            except:
                sort_1 = [data[i]] + sort_1
            break
        elif ind==len(sort_1)-1:
            sort_1.append(data[i])
            break
        else:
            pass

        ind+= 1

for i in sort_1:
    print(i['creationDate'])
print(len(sort_1))

### moving on
use = np.zeros(len(sort_1))
for i in range(0,len(sort_1)-1):
    pointA = sort_1[i]['qasms'][0]['qasm']
    pointB = sort_1[i+1]['qasms'][0]['qasm']
    if sort_1[i]['creationDate'][8:13]=='01T23':
        pass
    elif pointA==pointB:
        sort_2.append(sort_1[i])
        sort_2.append(sort_1[i+1])
  

for i in range(0,round(len(sort_2)/2)):
    pointA = sort_2[2*i]['qasms'][0]['qasm']
    pointB = sort_2[2*i+1]['qasms'][0]['qasm']
    print(pointA==pointB)

with open('e06_6p.dat','wb') as fp:
    pickle.dump(sort_2,fp,0)

print('Done!')
