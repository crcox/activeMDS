
import sys
#sys.path.append('../')
import activeMDS
from numpy import *
from numpy.random import *

norm = linalg.norm


def main():
    # load data
    S, label_lookup, query_type = read_triplets()
    n = len(label_lookup)

    # setup train/test splits
    S_TRAIN = []
    S_TEST = []
    for i in range(0,len(S)):
        if query_type[i] == 1:
            S_TRAIN.append(S[i])
        else:
            S_TEST.append(S[i])

    # desired dimension embedding
    d = 2
    
    # initialize embedding
    X = randn(n,d)
    X = X/norm(X)*sqrt(n)
    
    # number of times we will loop around the data to train the emebdding
    num_epochs = 30
    for i in range(0,num_epochs):
        
        X = activeMDS.update_embedding(S_TRAIN,X,len(S_TRAIN)*i,len(S_TRAIN)*(i+1))
        print "\n epoch = "+str((i+1))
        emp_loss,hinge_loss = activeMDS.get_loss(X,S_TRAIN)
        print "   TRAIN   emp_loss = "+str(emp_loss)+"   hinge_loss = "+str(hinge_loss)
        emp_loss,hinge_loss = activeMDS.get_loss(X,S_TEST)
        print "   TEST    emp_loss = "+str(emp_loss)+"   hinge_loss = "+str(hinge_loss)


#   d   TRAIN   TEST
#   1   28      29
#   2   24      28.5
#   3   21      31
#   4   19.5    30.5



def read_triplets():
    import csv
    import re
    
    ifile  = open('animal_responses.csv', "r")
    reader = csv.reader(ifile,escapechar='\\')
    
    print "starting read..."
    counter = 0;
    
    queryType0 = 0
    queryType1 = 0
    queryType2 = 0
    
    query_type = []
    Sl = []
    label_lookup = {}
    for row in reader:
        
#        if counter == 0:
#            for i in range(0,len(row)):
#                print str(i) + " : " + row[i]

        if counter > 0:
            q = [ row[7], row[9], row[5] ]
        
            label_lookup[row[7]] = 1
            label_lookup[row[9]] = 1
            label_lookup[row[5]] = 1
            
            if int(row[11])==0:
                queryType0 = queryType0 + 1
            if int(row[11])==1:
                queryType1 = queryType1 + 1
            if int(row[11])==2:
                queryType2 = queryType2 + 1
            
            Sl.append(q)
            query_type.append(int(row[11]))
        
        
        
        counter = counter + 1
    
    ifile.close()
    
    
    
    label2index = {}
    labels = []
    k = 0
    for x in label_lookup:
        label2index[x] = k
        labels.append(x)
        k = k + 1
    n = k
    
    #    print "n="+str(n)
    #print label2index

    S = []
    for ql in Sl:
        q = [ label2index[ql[0]], label2index[ql[1]], label2index[ql[2]] ]
        S.append(q)
    
    print "done reading! " + "n="+str(n) + "  |S|="+str(len(S))
    print "queryType0 = " + str(queryType0)
    print "queryType1 = " + str(queryType1)
    print "queryType2 = " + str(queryType2)
    return S, labels, query_type



if __name__ == "__main__":
    main()

