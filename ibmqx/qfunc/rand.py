import sys
import os
import datetime

def check(string1,string2):
    checker=False
    while checker==False:
        store = input(string1)
        if (store=='Y' or store=='y' or store=='yes' or store=='Yes'):
            pass
            checker=True 
        elif (store=='N' or store=='n' or store=='no' or store=='No'):
            sys.exit(string2) 
        else:
            print('Please type yes (y) or no (n).')
    return

def check_pass(string1,string2):
    checker=False
    while checker==False:
        store = 'y'
        if (store=='Y' or store=='y' or store=='yes' or store=='Yes'):
            pass
            checker=True 
        elif (store=='N' or store=='n' or store=='no' or store=='No'):
            sys.exit(string2) 
        else:
            print('Please type yes (y) or no (n).')
    return

def get_time():
    cur = datetime.datetime.now()
    cur_time =  cur.strftime('%m') + cur.strftime('%d') + cur.strftime('%y') + '_' + cur.strftime('%H') + cur.strftime('%M') + cur.strftime('%S')
    return cur_time

def recursive(list_of_lists,ind,temp,check,hold):
    if len(list_of_lists)>1:
        for i in list_of_lists[0]:
            temp.append(i)
            recursive(list_of_lists[1:],ind,temp,check,hold)
            holder = 1
            for j in list_of_lists[1:]:
                holder*= len(j)
            ind += holder
            temp.pop()
    else:
        for e in list_of_lists[0]:
            temp.append(e)
            hold.append(temp[:])
            temp.pop()
            ind+= 1
    if ind==(check):
        
        return hold 

