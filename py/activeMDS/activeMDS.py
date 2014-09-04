import unittest
import numpy

class Gradient:
    def __init__(self):
        self.g = None
        self.eloss = None
        self.hloss = None

def modelCheck(model, query):
    d1 = numpy.sqrt(numpy.sum(numpy.square(model[query[0],] - model[query[2],])))
    d2 = numpy.sqrt(numpy.sum(numpy.square(model[query[1],] - model[query[2],])))
    return(d1<d2)

def updateModel(model, query):
    def StochasticGradient(model, query):
        dims = numpy.ndarray.size(model)[1]
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        Xq = model[query,:]

        gradient = Gradient()
        gradient.g = zeros( (3,dims) )
        gradient.eloss = 0
        gradient.hloss = 0

        loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
        if loss_ijk + 1. > 0:
            gradient.hloss = loss_ijk + 1.
            gradient.g = gradient.g + H * Xq

            if loss_ijk > 0:
                emp_loss = 1.

        return gradient

    def regularize(model, query, regularization=10):
        for item in query:
            norm_i = norm(model(item, :))
            if norm_i > regularization:
                model[item, :] = model[item, :] * regularization / norm_i

        return model

    stepSize  = float(sqrt(100))/sqrt((self.total_epochs * self.query_count) + i + 100)
    gradient = StochasticGradient(model, query)
    model[query,:] = model[query,:] - (stepSize * Gq)  # max-norm
    #model = regularize(model, query)

    return model



class FooTests(unittest.TestCase):
    def testModelIsCorrect(self):
        primary = 0
        alternate = 1
        target = 2
        MODEL = numpy.mat([[1.0,1.0,1.0],[2.0,2.0,2.0],[0.0,0.0,0.0]])
        QUERY = [primary,alternate,target]
        self.failUnless(modelCheck(MODEL,QUERY))

    def testModelImproves(self):
        primary = 0
        alternate = 1
        target = 2
        MODEL = numpy.mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0])
        QUERY = [primary,alternate,target]
        self.failUnless(updateModel(MODEL,QUERY))

def main():
    unittest.main()

if __name__ == "__main__":
    main()
