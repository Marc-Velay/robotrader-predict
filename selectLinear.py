
def getLinear(values) :
    i = 0
    final = []
    j = values[i]
    aux = []
    comp = 0
    print (len(values))
    while i < (len(values)-1) :
        if(checkDist(j,values[i+1])) :
            aux.append(values[i+1])
        else:
            if(len(aux)>30) :
                final.append(aux)
            else:
                comp +=len(aux)
            aux = []
            j = values[i]
        i+=1
    if(len(aux)>30) :
        final.append(aux)
    else:
        comp +=len(aux)
    return (final,comp)

def checkDist(ref, todo) :
    return (not (int(todo[0])>(int(ref[0]) + (int(ref[0])*2)/100) or int(todo[0]) <(int(ref[0]) - (int(ref[0])*2)/100))) and (not (int(todo[1])>(int(ref[1]) + (int(ref[1])*2)/100) or int(todo[1]) <(int(ref[1]) - (int(ref[1])*2)/100)))
        