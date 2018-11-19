a = [1]
b = [4]
c = [7]
d = [12]
e = [0]
f = [0]
def rec(list_of_lists,depth,hold):
    for i in list_of_lists[-1]:
        depth.append(i)
        print(i)
        try:
            rec(list_of_lists[:-1],depth,hold)
        except IndexError:
            # can't fgo further in
            print('first error')
            pass
        depth = depth[:-1]
        print(depth)

def rec_2(list_of_lists,ind,temp,check,hold):
    print('Step 1')
    if len(list_of_lists)>1:
        for i in list_of_lists[0]:
            #print(i)
            temp.append(i)
            rec_2(list_of_lists[1:],ind,temp,check,hold)
            holder = 1
            for j in list_of_lists[1:]:
                holder*= len(j)
            ind+= holder
            temp.pop()
    else: # we have only 1 list left
        for element in list_of_lists[0]:
            temp.append(element)
            print(temp)
            hold.append(temp[:])
            temp.pop()
            print(ind)
            ind+=1
    if ind==(check):
        print('made it!')
        return hold
        
print(rec_2([a,b,c,d,e,f],0,[],1,[]))

#print(rec([a,b,c,d],[],[]))

