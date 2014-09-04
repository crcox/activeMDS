#from django.db import models
#from activeMDS.models import *
import csv
from itertools import islice
from numpy import dot,mat,sqrt,trace,zeros
from numpy.linalg import norm
from numpy.random import randn,shuffle
from math import ceil, floor

class Embedding:
    def __init__(self,data_path,d,regularization=10.0,proportion=1.0,train_codes=[1],test_codes=[0,2]):
    # N.B. Regularization constrains the size of the norm. 
    # 10.0 is sufficiently large to not matter. 
        def parseCodes(codes):
            try:
                codes_list = list(codes)
            except TypeError:
                codes_list = list([codes])
            return codes_list
   
        def determineType(codes):
            if 1 in codes:
                return "adaptive"
            else:
                return "random"
                
        self.d = d
        self.item_count = 0
        self.query_count = 0
        self.data_path = data_path
        self.proportion = proportion
        self.X = None
        self.S = None
        self.query_type = []
        self.query_type_count = [0, 0, 0]
        self.train_codes = parseCodes(train_codes)
        self.test_codes = parseCodes(test_codes)
        self.test_count = 0
        self.train_count = 0
        self.EmbeddingType = determineType(self.train_codes)
        self.labels = []
        self.total_epochs = 0
        self.error = {
            "emp_loss":[], 
            "emp_loss_cv":[],
            "hinge_loss":[],
            "hinge_loss_cv":[]
        }
        self.regularization = regularization
        
        # defines: S, lables, query_type, query_count, test_count, train_count
        self.read_triplets()
        
        # defines: X
        self.init_embedding()
    
    def read_triplets(self):    
        with open(self.data_path, 'rb') as ifile:
            if self.proportion < 1.0:
                num_lines = sum(1 for line in ifile)
                num_to_read = floor(num_lines * self.proportion)
                ifile.seek(0)
                reader = csv.reader(islice(ifile,num_to_read), escapechar='\\')
            else:
                reader = csv.reader(ifile,escapechar='\\')
        
            print "starting read..."    
            Sl = []
    
            header = reader.next() # reads first row
            primary = header.index('primary')
            alternate = header.index('alternate')
            target = header.index('target')
            qt = header.index('queryType')
    
            for row in reader: # reads rest of rows
                query = [ row[i] for i in (primary,alternate,target) ]
                query_code = int(row[qt])
                [ self.labels.append(x) for x in query if not x in self.labels ]
                Sl.append(query)
                self.query_type_count[query_code] += 1
                self.query_type.append(query_code)
                self.query_count += 1
                if query_code in self.test_codes:
                    self.test_count += 1
                else:
                    self.train_count += 1
                
        self.item_count = len(self.labels)
        self.labels.sort()
        label2index = dict( zip( self.labels,range(self.item_count) ) )
    
        self.S = []
        for queryLabels in Sl:
            q = [ label2index[x] for x in queryLabels ]
            self.S.append(q)
        
        print "done reading! " + "n="+str(self.item_count) + "  |S|="+str(len(self.S))
        print "queryType0 = " + str(self.query_type_count[0])
        print "queryType1 = " + str(self.query_type_count[1])
        print "queryType2 = " + str(self.query_type_count[2])
        return 0


    def init_embedding(self):
        self.X = randn(self.item_count,self.d)
        self.X = self.X/norm(self.X)*sqrt(self.item_count)
        return 0


    def update_embedding(self,nepochs=40,verbose=False):
        def print_iter():
            epoch_str = "epoch = {:2d}/{:2d}".format(self.total_epochs, startingEpoch+nepochs)
            emp_str = "emp_01_loss = {:.3f}".format(cum_emp_loss/self.train_count)
            hinge_str = "hinge_loss = {:.3f}".format(cum_hinge_loss/self.train_count)
            norm_str = "norm(X)/sqrt(n) = {:.3f}".format(norm(self.X)/sqrt(self.item_count))
            print '  '.join([epoch_str,emp_str,hinge_str,norm_str])
            return 0
            
        # regularization parameter is for the max-norm: if one of the l2-norm of
        # the rows exceeds regularization_lambda then project back
        startingEpoch = self.total_epochs
        firstTime = startingEpoch == 0

        # Select only the query types associated with the training set.
        S = [query for query,code in zip(self.S,self.query_type) if code in self.train_codes]     
 
        for epoch in range(nepochs):
            cum_emp_loss = 0
            cum_hinge_loss = 0
            shuffle(S)
            for i,query in enumerate(S):
                Gq,emp_loss,hinge_loss = self.get_stochasticGradient(query)
                cum_emp_loss = cum_emp_loss + emp_loss
                cum_hinge_loss = cum_hinge_loss + hinge_loss

                eta = float(sqrt(100))/sqrt((self.total_epochs * self.query_count) + i + 100)
                self.X[query,:] = self.X[query,:] - eta*Gq  # max-norm

                for item in query:
                    norm_i = norm(self.X[item,:])
                    if norm_i>self.regularization:
                        self.X[item,:] = self.X[item,:] * self.regularization / norm_i

            self.evaluateEmbedding()
            self.total_epochs += 1  

            if verbose:
                print_iter()

        # if start_iter>0 then we're doing something incremental and we 
        # should not normalize each time.
        if firstTime:  
            self.X = self.X/norm(self.X)*sqrt(self.item_count)

        return 0


    def write(self,prefix):
        error_by_epoch_file = "{}_e{}_p{}_d{}_{}_errByEpoch.csv".format(
            prefix,
            self.total_epochs,
            int(self.proportion * 100),
            self.d,
            self.EmbeddingType
        )
        with open(error_by_epoch_file,'wb') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'error',
                'error_cv',
                'hinge',
                'hinge_cv'
            ])
            ERR = zip(
                self.error["emp_loss"],
                self.error["emp_loss_cv"],
                self.error["hinge_loss"],
                self.error["hinge_loss_cv"]
            )
            for i,(e,ecv,h,hcv) in enumerate(ERR):
                writer.writerow([i,e,ecv,h,hcv])
				
        embedding_file = "{}_e{}_p{}_d{}_{}_embedding.csv".format(
            prefix,
            self.total_epochs,
            int(self.proportion * 100),
            self.d,
            self.EmbeddingType
        )
		
        with open(embedding_file,'wb') as f:
            writer = csv.writer(f)
            h = ['x'+str(i+1) for i in range(self.d)]
            writer.writerow(['label']+h)
            for lab,x in zip(self.labels,self.X):
                writer.writerow([lab]+x.tolist())

        query_file = "{}_e{}_p{}_d{}_{}_queries.csv".format(
            prefix,
            self.total_epochs,
            int(self.proportion * 100),
            self.d,
            self.EmbeddingType
        )
		
        with open(query_file,'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['type','primary','alterate','target'])
            for lab,x in zip(self.query_type,self.S):
                writer.writerow([lab]+x)
        return None


    def evaluateEmbedding(self):
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        emp_loss = 0
        emp_loss_cv = 0 # 0/1 loss
        hinge_loss = 0
        hinge_loss_cv = 0 # hinge loss
        for query,code in zip(self.S,self.query_type):
            Xq = self.X[query,:]
            loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
            if code in self.test_codes:
                if loss_ijk+1.>0.:
                    hinge_loss_cv += (loss_ijk + 1.0)
                    
                    if loss_ijk > 0:
                        emp_loss_cv += 1.0
            else:
                if loss_ijk+1.>0.:
                    hinge_loss += (loss_ijk + 1.0)
                    
                    if loss_ijk > 0:
                        emp_loss += 1.0
            
        self.error["emp_loss"].append(emp_loss/self.train_count)
        self.error["emp_loss_cv"].append(emp_loss_cv/self.test_count)
        self.error["hinge_loss"].append(hinge_loss/self.train_count)
        self.error["hinge_loss_cv"].append(hinge_loss_cv/self.test_count)

        return 0


    def get_stochasticGradient(self,query):
        # returns gradient wrt loss function loss(X,q)
        # returns 3-by-d vector that should update X at indices q

        Xq = self.X[query,:]

        emp_loss = 0 # 0/1 loss
        hinge_loss = 0 # hinge loss
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        G = zeros((3,self.d))
        loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
        if loss_ijk+1.>0.:
            hinge_loss = loss_ijk + 1.
            G = G + H * Xq
            
            if loss_ijk > 0:
                emp_loss = 1.
    
        return G, emp_loss, hinge_loss