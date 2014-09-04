import sys
import os
import csv
import activeMDS
from numpy import *
from numpy.random import *

norm = linalg.norm
floor = math.floor
ceil = math.ceil

default_dir = '/'

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


def read_triplets(f):
    import csv
    import re
    
    ifile  = open(f, "r")
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

def sandbox():
    
    z = [randint(100),randint(100),randint(100)]
    print z


def get_next_query(X,S=[]):
    # performs uncertainty sampling

    n = X.shape[0]
    d = X.shape[1]
    
    
    H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])

    num_candidates = 100
    
    score = inf
    q_opt = [-1, -1, -1]
    for iter in range(0,num_candidates):
    
        i = randint(n)
        j = randint(n)
        while (j==i):
            j = randint(n)
        k = randint(n)
        while (k==i) | (k==j):
            k = randint(n)
        q = [i, j, k]

        temp = abs(trace(dot(H,dot(X[q,:],X[q,:].T))))
        if temp < score:
            q_opt = [i, j, k]
            score = temp

    return q_opt


def plot_embedding(X,Xtrue):
    n = X.shape[0]
    d = X.shape[1]
    
    import matplotlib
    import matplotlib.pyplot as plt
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(Xtrue[:,0], Xtrue[:,1], 'bo')
    for k in range(0,n):
        plt.text(Xtrue[k,0], Xtrue[k,1], str(k))
    
    #    plt.axis([0, 6, 0, 20])
    
    plt.subplot(212)
    plt.plot(X[:,0], X[:,1], 'ro')
    for k in range(0,n):
        plt.text(X[k,0], X[k,1], str(k))
    plt.show()



def update_embedding(S,X,start_iter=0,end_iter=nan):
    
    n = X.shape[0]
    d = X.shape[1]
    m = len(S)
    if isnan(end_iter):
        end_iter = 20*m
    
    count = 0
    avg_emp_loss = 0
    avg_hinge_loss = 0
    random_permutation = range(0,m)
    shuffle(random_permutation)
    for iter in range(start_iter,end_iter):
        #        G,avg_emp_loss,avg_hinge_loss = get_gradient(X,S)
        #        eta = 400.
        
        q = S[random_permutation[count]]
        G,emp_loss,hinge_loss = get_gradient(X,[q])
        avg_emp_loss = avg_emp_loss + emp_loss
        avg_hinge_loss = avg_hinge_loss + hinge_loss
        count = count + 1
        eta = float(sqrt(100))/sqrt(iter+100)
        
        X = X - eta*G
        
        if iter % m == 0:
            
#            print "epoch = "+str(iter/m)+"   emp_loss = "+str(avg_emp_loss/count)+"   hinge_loss = "+str(avg_hinge_loss/count)+"    norm(X)/sqrt(n) = "+str(norm(X)/sqrt(n))
            avg_emp_loss = 0
            avg_hinge_loss = 0
            count = 0
            shuffle(random_permutation)

    return X/norm(X)*sqrt(n)


def get_loss(X,S):
    # For loss in 0/1 or hinge, returns empirical loss 1/m sum_{ell = 1}^m loss(X,S[ell,:])

    n = X.shape[0]
    d = X.shape[1]
    m = len(S)*1.

    emp_loss = 0 # 0/1 loss
    hinge_loss = 0 # hinge loss

    # S[iter,:] = [i,j,k]   <=>    norm(xi-xk)<norm(xj-xk) )
    H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
    
    for q in S:
        loss_ijk = trace(dot(H,dot(X[q,:],X[q,:].T)))
        if loss_ijk+1.>0.:
            hinge_loss = hinge_loss + (loss_ijk + 1.)
            
            if loss_ijk > 0:
                emp_loss = emp_loss + 1.

    emp_loss = emp_loss/m
    hinge_loss = hinge_loss/m

    return emp_loss, hinge_loss


def get_gradient(X,S):
    # returns gradient wrt loss function 1/m sum_{ell = 1}^m loss(X,S[ell,:])
    
    n = X.shape[0]
    d = X.shape[1]
    m = len(S)*1.
    
    emp_loss = 0 # 0/1 loss
    hinge_loss = 0 # hinge loss
    
    # S[iter,:] = [i,j,k]   <=>    norm(xi-xk)<norm(xj-xk) )
    H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
    
    G = zeros((n,d))
    for q in S:
        loss_ijk = trace(dot(H,dot(X[q,:],X[q,:].T)))
        if loss_ijk+1.>0.:
            hinge_loss = hinge_loss + loss_ijk + 1.
            G[q,:] = G[q,:] + H*X[q,:]/m
            
            if loss_ijk > 0:
                emp_loss = emp_loss + 1.
    

    emp_loss = emp_loss/m
    hinge_loss = hinge_loss/m
    
    return G, emp_loss, hinge_loss


if __name__ == "__main__":
    main()

