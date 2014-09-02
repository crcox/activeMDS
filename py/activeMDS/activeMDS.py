import unittest
import numpy

def modelCheck(model, query):
    d1 = numpy.sqrt(numpy.sum(numpy.square(model[query[0],] - model[query[2],])))
    d2 = numpy.sqrt(numpy.sum(numpy.square(model[query[1],] - model[query[2],])))
    return(d1<d2)

def updateModel(model, query):
    

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
