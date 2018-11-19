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
    try:
        check4 = (res[0]['status']=='DONE' and res[1]['status']=='DONE')
    except:
        pass
    time = item['creationDate'][0:]
    hr = time[2:4] 
    mn = time[5:7]
    sc = time[8:10]
    time = [hr,mn,sc]
    check1 = (earlier(time,['18','02','15']))
    if not check1:
        print(check1,time)


