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

with open('ibm_data.dat','rb') as fp:
    data = pickle.load(fp)
i = 0
hold = []
dats = []
for item in data:
    res = item['qasms']
    check1 = (item['creationDate'][0:10]=='2018-01-12' or item['creationDate'][0:10]=='2018-01-12')
    check2 = (item['backend']['name']=='ibmqx4')
    check3 = (int(item['creationDate'][11:13])>6)
    if (check1 and check2 and check3):
        check4 = (res[0]['status']=='DONE' and res[1]['status']=='DONE')
        if check4:
            time = item['creationDate'][10:]
            hr = time[1:3] 
            mn = time[4:6]
            sc = time[7:9]
            time = [hr,mn,sc]
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
for i in hold:
    print(i)
print(dats)
with open('e04.dat','wb') as fp:
    pickle.dump(dats,fp,0)


