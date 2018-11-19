import pickle
import numpy
import qiskit

###
# data keys:
# shots,backend,maxCredits,deleted,qasms,userId,calibrations,status,creationDate,usedCredits,id
###
def earlier(one,two):
    #hr, min, sec
    # is one earlier than two?
    if int(one[0])<int(two[0]):
        return True
    elif int(one[0])==int(two[0]):
        if int(one[1])<int(two[1]):
            return True
        elif int(one[1])==int(two[1]):
            if int(one[2])<int(two[2]):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
test = ['22','49','07']

with open('ibm_exp_01_04.dat','rb') as fp:
    data = pickle.load(fp)
i = 0
hold = []
dats = []
for item in data:
    res = item['qasms']
    check2 = (item['backend']['name']=='ibmqx2')
    if (check2):
        time = item['creationDate'][10:]
        hr = time[1:3]
        mn = time[4:6]
        sc = time[7:9]
        time = [hr,mn,sc]
        if earlier(time,[8,24,0]):
            continue
        else:
            pass
        if i==0:
            hold.append(time)
            dats.append(item)
            i+=1 
            continue
        early = False
        for z in range(0,len(hold)):
            check = earlier(time,hold[z])
            if check:
                early=True
            else:
                pass
            if early:
                hold=hold[0:z]+[time]+hold[z:]
                dats=dats[0:z]+[item]+dats[z:]
                break
            if z==len(hold)-1:
                hold.append(time)
                dats.append(item)
                break
        i+=1 

print(len(hold))

e01 = dats[0:118]
e04 = dats[118:]


print(dats[116]['qasms'][0]['qasm'])
print(dats[117]['qasms'][0]['qasm'])
print(dats[118]['qasms'][0]['qasm'])
print(dats[119]['qasms'][0]['qasm'])
print(dats[120]['qasms'][0]['qasm'])
print(dats[121]['qasms'][0]['qasm'])

with open('e01_6p.ibm.dat','wb') as fp:
    pickle.dump(e01,fp,0)
with open('e04_3p.ibm.dat','wb') as fp:
    pickle.dump(e04,fp,0)


