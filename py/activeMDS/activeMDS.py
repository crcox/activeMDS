import unittest
import csv
from itertools import count
from math import floor
from numpy import dot,mat,sqrt,trace,zeros,sum,ndarray,square
from numpy.linalg import norm
from numpy.random import randn,shuffle

master_json = {
    'nepochs': 10,
    'traincodes': [1],
    'verbose': True,
    'log': {'prefix':'test'},
    'data':'/Users/chris/activeMDS/example/test.csv'
}
config_json = {
    'proportion': 1.0,
    'regularization': 4 # <= 0 is undefined, 10 is essentially a ceiling
}


def updateModel(model,query,STEP_COUNTER):
    def StochasticGradient(Xq):
        dims = model.shape[1]
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])

        g = zeros( (3,dims) )
        emploss = 0
        hingeloss = 0

        loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
        if loss_ijk + 1. >= 0:
            hingeloss = loss_ijk + 1.
            g = g + H * Xq

            if loss_ijk >= 0:
                emploss = 1.

        return g, emploss, hingeloss, loss_ijk

    def regularize(Xq, regularization=10):
        for i in range(3):
            norm_i = norm(Xq[i, :])
            if norm_i > regularization:
                Xq[i, :] *= (regularization / norm_i)

        return Xq

    # Select the model elements relevant to the current query.
    Xq = model[query, :]
    stepSize  = sqrt(100.)/sqrt(next(STEP_COUNTER) + 100.)
    g,emploss,hingeloss,loss = StochasticGradient(Xq)
    Xq -= (stepSize * g)  # max-norm
    #Xq = regularize(Xq)
    model[query, :] = Xq

    return {'gradient':g,'emploss':emploss,'hingeloss':hingeloss,'loss':loss}

def evaluateModel(model,query,STEP_COUNTER):
    def computeLoss(Xq):
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
        emploss = 0
        hingeloss = 0
        if loss_ijk + 1. >= 0:
            hingeloss = loss_ijk + 1.
            if loss_ijk >= 0:
                emploss = 1.
        return emploss, hingeloss, loss_ijk

    # Select the model elements relevant to the current query.
    Xq = model[query, :]
    emploss,hingeloss,loss = computeLoss(Xq)

    return {'emploss':emploss,'hingeloss':hingeloss,'loss':loss}

def read_triplets(data_path):
    with open(data_path, 'rb') as ifile:
        reader = csv.reader(ifile,escapechar='\\')

        print "starting read..."
        Sl = []

        header = reader.next() # reads first row
        primary = header.index('primary')
        alternate = header.index('alternate')
        target = header.index('target')
        qt = header.index('queryType')
        labels = []
        query_count = 0
        query_type = []
        query_type_count = {}
        for row in reader: # reads rest of rows
            query = [ row[i].strip() for i in (primary,alternate,target) ]
            query_code = int(row[qt].strip())
            [ labels.append(x.strip()) for x in query if not x in labels ]
            Sl.append(query)
            try:
                query_type_count[query_code] += 1
            except KeyError:
                query_type_count[query_code] = 1

            query_type.append(query_code)
            query_count += 1

    item_count = len(labels)
    labels.sort()

    S = []
    for queryLabels in Sl:
        q = [ labels.index(x) for x in queryLabels ]
        S.append(q)

    print "done reading! " + "n="+str(item_count) + "  |S|="+str(len(S))
    print "queryType0 = " + str(query_type_count[0])
    print "queryType1 = " + str(query_type_count[1])
    print "queryType2 = " + str(query_type_count[2])

    return {'queries': S,
            'querytype': query_type,
            'nitems': item_count,
            'nqueries': query_type_count,
            'labels': labels
            }

def writeModel(model, labels, opts):
    with open(opts['modelfilename'], 'wb') as f:
        writer = csv.writer(f,escapechar='\\')
        writer.writerows(zip(labels,model))

def writeLoss(lossLog, opts):
    with open(opts['lossfilename'], 'wb') as f:
        writer = csv.writer(f,escapechar='\\')
        writer.writerow(['eloss','hloss','eloss_t','hloss_t'])
        writer.writerows(lossLog)

def initializeEmbedding(nitems, dimensions):
    model = randn(nitems,dimensions)
    model = model/norm(model)*sqrt(nitems)
    return model

def fitModel(model, responses, opts=False):
    STEP_COUNTER = count(1)
    def printLoss(MDATA):
        epoch_str = "epoch = {:2d}".format(MDATA['epoch'])
        emp_str = "emploss = {:.3f}".format(MDATA['emploss'])
        hinge_str = "hingeloss = {:.3f}".format(MDATA['hingeloss'])
        norm_str = "norm(X)/sqrt(n) = {:.3f}".format(MDATA['norm'])
        print '  '.join([epoch_str,emp_str,hinge_str,norm_str])

    if not opts:
        opts = {
            'proportion': 1.0,
            'nepochs': 10,
            'traincodes': [1],
            'verbose': True,
            'log': True,
            'debug': False
        }

    lastQuery = int(floor(sum(responses['nqueries'].values()) * opts['proportion']))
    ntest = 0
    ntrain = 0
    for k,v in responses['nqueries'].items():
        if k in opts['traincodes']:
            ntrain += v
        else:
            ntest += v

    S = [(query,code) for query,code in zip(responses['queries'],responses['querytype'])]
    S = S[:lastQuery]
    lossLog = []
    for epoch in range(opts['nepochs']):
        MDATA = {'emploss': 0, 'hingeloss': 0, 'epoch': epoch}
        MDATA_test = {'emploss': 0, 'hingeloss': 0, 'epoch': epoch}
        shuffle(S)
        for i,(query,code) in enumerate(S):
            if code in opts['traincodes']:
                QDATA = updateModel(model, query, STEP_COUNTER)
                MDATA['emploss'] += QDATA['emploss']
                MDATA['hingeloss'] += QDATA['hingeloss']
            else:
                QDATA = evaluateModel(model, query, STEP_COUNTER)
                MDATA_test['emploss'] += QDATA['emploss']
                MDATA_test['hingeloss'] += QDATA['hingeloss']


        if opts['verbose'] == True:
            MDATA['emploss'] /= float(ntrain)
            MDATA['norm'] = norm(model) / sqrt(responses['nitems'])
            printLoss(MDATA)

        if opts['log'] == True:
            lossLog.append(
                    [
                        MDATA['emploss']/float(ntrain),
                        MDATA['hingeloss'],
                        MDATA_test['emploss']/float(ntest),
                        MDATA_test['hingeloss']
                    ]
                )

    if opts['debug']:
        return len(S)
    else:
        return lossLog

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.primary = 0
        self.alternate = 1
        self.target = 2

    def testModelIsCorrect(self):
        def modelCheck(model, query):
            d1 = sqrt(sum(square(model[query[0],] - model[query[2],])))
            d2 = sqrt(sum(square(model[query[1],] - model[query[2],])))
            return(d1<d2)

        MODEL = mat([[1.0,1.0,1.0],[2.0,2.0,2.0],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        self.failUnless(modelCheck(MODEL,QUERY))

    def testModelImproves(self):
        MODEL = mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        MODEL_orig = MODEL.copy()
        QUERY = [self.primary,self.alternate,self.target]
        # Remember that lists/numpy arrays are mutable, so they are updated in place.
        # updateModel() will update MODEL without needing to return a new model.
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(sum(MODEL[0,:]) < sum(MODEL_orig[0,:]))

    def testRight_eloss(self):
        # Right
        MODEL = mat([[1.0,1.0,1.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 0)

    def testWrong_eloss(self):
        # Wrong
        MODEL = mat([[2.0,2.0,2.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 1)

    def testEqualIsWrong_eloss(self):
        # Wrong
        MODEL = mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 1)

    def testRight_hloss(self):
        # Right
        MODEL = mat([[1.0,1.0,1.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] == 0)

    def testWrong_hloss(self):
        # Wrong
        MODEL = mat([[2.0,2.0,2.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] > 1)

    def testEqualIsWrong_hloss(self):
        # Wrong
        MODEL = mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] == 1)

    def testStepCounter(self):
        def a(c):
            for i in range(10):
                n = next(c)
            return n
        STEP_COUNTER = count(1)
        for i in range(10):
            n = a(STEP_COUNTER)
        print n
        self.failUnless(n==100)

class IOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Remember that lists are mutable, and so updates to model that take
        # place within fit model will have affects beyond the scope of the
        # function.
        data_path = '/Users/Chris/activeMDS/example/test.csv'
        cls.responses = read_triplets(data_path)
        cls.model = initializeEmbedding(cls.responses['nitems'],3)
        cls.lossLog = fitModel(cls.model, cls.responses)
        with open(data_path, 'rb') as ifile:
            reader = csv.reader(ifile,escapechar='\\')
            header = reader.next() # reads first row
            primary = header.index('primary')
            alternate = header.index('alternate')
            target = header.index('target')
            cls.check = []
            for row in reader: # reads rest of rows
                cls.check.append([ row[i].strip() for i in (primary,alternate,target) ])

    def testRead_nitems(self):
        self.failUnless(self.responses['nitems']==7)

    def testRead_nqueries(self):
        check = {0:2, 1:2, 2:2}
        match = [check[key] == val for key,val in self.responses['nqueries'].items()]
        self.failUnless(all(match))

    def testRead_queryConstruction(self):
        q = [[self.responses['labels'][i] for i in indexes] for indexes in self.responses['queries']]
        self.failUnless(q == self.check)

    def testWrite_model(self):
        writeModel(self.model,self.responses['labels'],{'modelfilename':'model.csv'})
        self.failUnless(True)

    def testWrite_loss(self):
        writeLoss(self.lossLog,{'lossfilename':'loss.csv'})
        self.failUnless(True)

    def test_proportion(self):
        opts = {
            'proportion': 0.5,
            'nepochs': 10,
            'traincodes': [1],
            'verbose': True,
            'log': True,
            'debug':True
        }
        n = fitModel(self.model, self.responses, opts)
        self.failUnless(n==3)

# class ImportantTests(unittest.TestCase):
#     def test_queryCheck(self):
#         """This confirms that the internal coding scheme for items exactly
#         maps to the original data."""
#         with open('/Users/Chris/activeMDS/example/test.csv', 'rb') as ifile:
#             reader = csv.reader(ifile,escapechar='\\')
#             header = reader.next() # reads first row
#             primary = header.index('primary')
#             alternate = header.index('alternate')
#             target = header.index('target')
#             queries = []
#             for row in reader: # reads rest of rows
#                 queries.append([ row[i].strip() for i in (primary,alternate,target) ])
#
#         q = [[self.responses['labels'][i] for i in indexes] for indexes in self.responses['queries']]
#         self.failUnless(q == queries)

def main():
    unittest.main()

if __name__ == "__main__":
    main()
