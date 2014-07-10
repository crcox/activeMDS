#from django.db import models
#from activeMDS.models import *
import os
import csv
from numpy import *
from numpy.random import *

norm = linalg.norm
floor = math.floor
ceil = math.ceil

default_dir = '/'

def sandbox():
    
    z = [randint(100),randint(100),randint(100)]
    print z


def main():
    
    
    # generate some fake data
    n = 30
    d = 2
    total_num_queries = int(ceil(10*n*d*log(n)))  # number of labels
    total_num_test_queries = 1000;
    
    p = 0.1; # error rate
    
    S_test = []
    Xtrue = randn(n,d);
    for iter in range(0,total_num_test_queries):
        i = randint(n)
        j = randint(n)
        while (j==i):
            j = randint(n)
        k = randint(n)
        while (k==i) | (k==j):
            k = randint(n)
        q = [i, j, k]
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        query_ordering_incorrect = trace(dot(H,dot(Xtrue[q,:],Xtrue[q,:].T)))>0
        if query_ordering_incorrect:
            q = [ q[i] for i in [1,0,2]]
        if rand()<p:
            q = [ q[i] for i in [1,0,2]]

        S_test.append(q)





    # initialize embedding
    S = []
    X = randn(n,d)
    X = X/norm(X)*sqrt(n)

    
    for iter in range(0,total_num_queries):
    
        # get candidate query
        q = get_next_query(X)
        
        # get noisy label
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        query_ordering_incorrect = trace(dot(H,dot(Xtrue[q,:],Xtrue[q,:].T)))>0
        if query_ordering_incorrect:
            q = [ q[i] for i in [1,0,2]]
        if rand()<p:
            q = [ q[i] for i in [1,0,2]]

        # add observed relation to the set
        S.append(q)
        
#        print S
#        temp = raw_input()

        # update embedding
        X = update_embedding(S,X,30*len(S),len(S)*31)
    
        # check loss
        emp_loss,hinge_loss = get_loss(X,S_test)
        print "iter = "+str(iter)+"   emp_loss = "+str(emp_loss)+"   hinge_loss = "+str(hinge_loss)
    

#    plot_embedding(X,Xtrue)





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








#
#
#
#
#
#
#
#
#def setup_experiment_script(exp):
#    prj_id = exp.project.id
#    n = exp.num_objects
#    d = exp.target_dimension
#
#    params = {'n':n,'d':d,'gamma': .1, 'Ntrain':1000,'iter': 0,'thresh':.01/numpy.power(n,.75)}
#    X = numpy.random.normal(0,1,(n,d))
#    norm_X = numpy.linalg.norm(X)
#    if norm_X > 1:
#        X = X / norm_X
#    
#    G = numpy.zeros((n,d))
#
#    write_stuff(prj_id,X,G,params)
#
#
#    
##    exp_list = [exp]
##    for child_exp in exp.project.experiment_set.all():
##        if child_exp.epoch < exp.epoch:
##            exp_list.append(child_exp)
##
##    first_exp = exp
##    first_exp_id = 100000
##    for exp_temp in exp_list:
##        if exp_temp.id < first_exp_id:
##            first_exp_id = exp_temp.id
##            first_exp = exp_temp
##
##    num_labeled_data = 0
##    num_unlabeled_data = 0    
##    for exp_temp in exp_list:
##        for ses in exp_temp.session_set.all():
##            for que in ses.query_set.all():
##                if exp_temp.id==first_exp_id:
##                    num_labeled_data = num_labeled_data + 1
##                    num_unlabeled_data = num_unlabeled_data + que.unlabeled_query_count
##                elif que.query_type==2:
##                    num_labeled_data = num_labeled_data + 1
##                    num_unlabeled_data = num_unlabeled_data + que.unlabeled_query_count
##
##    num_data_total = num_unlabeled_data + num_labeled_data
##
##    execute_string = 'echo num_unlabeled_data'+str(num_unlabeled_data)
##    os.system(execute_string)
##    execute_string = 'echo num_labeled_data'+str(num_labeled_data)
##    os.system(execute_string)
##
##    if num_data_total > 0:
##        # let Rhat(w) - R(w) < agnostic_excess_risk_bound
##        agnostic_excess_risk_bound = numpy.sqrt(  (2*d*n*numpy.log(n))/(2*num_data_total)  )
##        exp.query_threshold = numpy.pi/n * agnostic_excess_risk_bound
##
##        write_queries( exp_list )
##        generate_embedding(exp.target_dimension)
##        X = read_embedding()
##    else:
##        exp.query_threshold = 2*numpy.pi/n
##        X = numpy.random.normal(0,1,(exp.num_objects,exp.target_dimension));
##    
##
##
##    X = X / numpy.linalg.norm(X)
##    temp = X.reshape(n*d).copy() # writes like [ X(1,1), X(1,2), ... , X(1,d), X(2,1) ... ]
##    exp.embedding_string = ','.join([str(i) for i in temp])
##    exp.save()
#
#
#
#
#
#def read_stuff(prj_id):
#    
#    directory = '/home/website/emoWords/activeMDS/static/activeMDS/'+str(prj_id)+'/'
#    
#    with open(directory+'active_params.txt', 'r') as csvfile:
#        reader = csv.reader(csvfile)
#        params = dict(x for x in reader)
#
#    n = int(params['n'])
#    d = int(params['d'])
#
#    
#    
#    # embedding and written data has NO indices. Straight n-by-d matrices
#    X = numpy.genfromtxt(directory+'active_embedding.txt')
#    X = X.reshape(n,d)
#    G = numpy.genfromtxt(directory+'active_gradient.txt')
#    G = G.reshape(n,d)
#
#    return X, G, params
#
#def write_stuff(prj_id,X,G,params):
#    n = int(params['n'])
#    d = int(params['d'])
#    
#    directory = '/home/website/emoWords/activeMDS/static/activeMDS/'+str(prj_id)+'/'
#
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    
#    with open(directory+'active_params.txt', 'w') as csvfile:
#        writer = csv.writer(csvfile)
#        for key, value in params.items():
#            writer.writerow([key, value])
#
#    X = X.reshape(n*d)
#    numpy.savetxt(directory+'active_embedding.txt',X)
#    G = G.reshape(n*d)
#    numpy.savetxt(directory+'active_gradient.txt',G)
#
#    
#
#
#def get_query(exp,que_type):
#    
#    prj_id = exp.project.id
#    n = exp.num_objects
#    if que_type in [0,1]:
#        i=numpy.random.randint(n)
#        j=numpy.random.randint(n)
#        while i==j:
#            j=numpy.random.randint(n)
#        k=numpy.random.randint(n)
#        while k in [i,j]:
#            k=numpy.random.randint(n)
#        return Query(query_answer=-1,response_time=-1,query_type=que_type,center=k+1,left=i+1,right=j+1,unlabeled_query_count=0)
#    elif que_type==2:
#        
#        
#        
#        X, G, params = read_stuff(prj_id)
#        t = int(params['iter'])
#        thresh = float(params['thresh'])
#        
#        count = 0
#        is_bad_query = True
#        while is_bad_query:
#            count = count + 1
#            i=numpy.random.randint(n)
#            j=numpy.random.randint(n)
#            while i==j:
#                j=numpy.random.randint(n)
#            k=numpy.random.randint(n)
#            while k in [i,j]:
#                k=numpy.random.randint(n)
#            
#            dec_var = abs(numpy.dot(X[i],X[i]) - 2*numpy.dot(X[i],X[k]) - numpy.dot(X[j],X[j]) +2*numpy.dot(X[j],X[k]))
#            
#            if dec_var < thresh:
#                is_bad_query = False
#            if count > 10000:
#                os.system('echo wtf happened in get_query?')
#                break
#                
#        
#
#        return Query(query_answer=-1,response_time=-1,query_type=que_type,center=k+1,left=i+1,right=j+1,unlabeled_query_count=count)
#
#
#
#def update_embedding(exp,i,j,k,answer):
#    
#    prj_id = exp.project.id
#    
#    # answer = 0 if left is chosen, 1 if right chosen. Need to convert to yt=sign( norm(xj-xk)-norm(xi-xk) )
#    if answer==0:
#        yt = 1.
#    else:
#        yt = -1.
#
#    # BE FUCKING CAREFUL ABOUT INDICES. i,j,k start at 1, NOT 0.
#    q = [int(i)-1,int(j)-1,int(k)-1]
#    Ht = numpy.mat([[-1.,0.,1.],[ 0.,  1.,  -1.],[ 1.,  -1.,  0.]])
#    
#    X, G, params = read_stuff(prj_id)
#    t = int(params['iter'])
#    gamma = float(params['gamma'])
#
#    t = t + 1
##    G[numpy.ix_(q,q)] = G[numpy.ix_(q,q)] - yt*Ht
#    G[q,:] = G[q,:] - yt*Ht*X[q,:]
#
##    X = X - 2.*gamma*numpy.dot(G,X)/numpy.linalg.norm(G)/numpy.sqrt(t)
#    X = X - 2.*gamma*G/numpy.sqrt(t)
#
#    norm_X = numpy.linalg.norm(X)
#    if norm_X > 1:
#        X = X / norm_X
#
#    params['iter'] = t
#
#    write_stuff(prj_id,X,G,params)
#
#
#
#
#
#
#
#
#
#
#
#
#
##for exp in Experiment.objects.all():
##   exp_list.append(exp)
#def write_queries(exp_list,que_type_for_embedding=[2]):
#    # writes out queries associated with all experiments in exp_list
#    
#    first_exp_id = 100000
#    for exp in exp_list:
#        if exp.id < first_exp_id:
#            first_exp_id = exp.id
#    
#    file_name = '/home/website/emoWords/database/relations_file.txt'
#    f = open(file_name, 'w')
#    # f.readline()
#    #for line in f:
#    #   print line,
#    # f.write('This is a test\n')
#    for exp in exp_list:
#        for ses in exp.session_set.all():
#            for que in ses.query_set.all():
#                if exp.id==first_exp_id and exp.project.id<=2:
#                    line = ''
#                    if que.query_answer==0:
#                        line = str(que.left)+','+str(que.right)+','+str(que.center)+'\n'
#                    elif que.query_answer==1:
#                        line = str(que.right)+','+str(que.left)+','+str(que.center)+'\n'
#                    #
#                    f.write(line)
#                elif que.query_type in que_type_for_embedding:
#                    #                if que.query_type in que_type_for_embedding:
#                    line = ''
#                    if que.query_answer==0:
#                        line = str(que.left)+','+str(que.right)+','+str(que.center)+'\n'
#                    elif que.query_answer==1:
#                        line = str(que.right)+','+str(que.left)+','+str(que.center)+'\n'
#                    
#                    f.write(line)
#    
#    f.close()
#
#
#def read_embedding():
#    
#    #    X = numpy.zeros((exp.num_objects,exp.target_dimension))
#    #    with open('/home/website/emoWords/database/embedding.txt') as input_file:
#    #        for line in input_file:
#    #            line = line.strip()
#    #            k=0
#    #            for number in line.split():
#    #                if k==0:
#    #                    index = int(number)
#    #                else:
#    #                    X[index-1][k-1] = float(number)
#    #                k=k+1
#    #
#    #    input_file.close()
#    data = numpy.genfromtxt('/home/website/emoWords/database/embedding.txt')
#    I=numpy.argsort(data[:,0])
#    X = data[I,:]
#    X = X[:,1:numpy.shape(X)[1]]
#    
#    return X
#
#
#def generate_embedding(target_dimension=2):
#    
#    os.system('g++ -O3 /home/website/emoWords/activeMDS/fastMDS.cpp')
#    execute_string = './a.out '+str(target_dimension)
#    os.system(execute_string)
#


if __name__ == "__main__":
    main()
