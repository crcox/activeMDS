import unittest
import datetime

class DatePattern:
    def __init__(self, year, month, day):
        pass

    def matches(self, date):
        return True

class FooTests(unittest.TestCase):
    def testMatches(self):
        p = DatePattern(2004, 9 ,28)
        d = datetime.date(2004, 9, 28)
        self.failUnless(p.matches(d))
    def testMatchesFalse(self):
        p = DatePattern(2004, 9 ,28)
        d = datetime.date(2004, 9, 28)
        self.failIf(p.matches(d))

def main(): 
    unittest.main()

if __name__ == '__main__':
    main()
