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
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        dims = numpy.ndarray.shape(model)[1] # number of dimensions
        X = model[query, ]
        gradient = Gradient()
        gradient.g = zeros((3,dims))
        gradient.eloss = 0
        gradient.hloss = 0




    gradient = StochasticGradient()


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
