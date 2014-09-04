import activeMDS
import argparse
import numpy.random

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('d', metavar="dimensions", type=int, 
        help='an integer specifying the number of dimensions in which to embed the data.')
parser.add_argument('n_epochs', metavar="n epochs", type=int,
        help='an integer specifying the number of iterations the optimization should take.')
parser.add_argument('infile', metavar='data', type=str,
        help='path to input data; output from NEXT.Discovery.')
parser.add_argument('outfile', metavar='embedding', type=str,
        help='output filename. If not path is specified, will write to current directory.')

args = parser.parse_args()
print(args)


# load data
S, label_lookup, query_type = activeMDS.read_triplets(args.infile)
n = len(label_lookup)

# setup train/test splits
S_TRAIN = []
S_TEST = []
for i in range(0,len(S)):
    if query_type[i] == 1:
        S_TRAIN.append(S[i])
    else:
        S_TEST.append(S[i])

# initialize embedding
X = numpy.random.randn(n,args.d)
X = X/numpy.linalg.norm(X)*numpy.math.sqrt(n)

# number of times we will loop around the data to train the emebdding
for i in range(0,args.n_epochs):
    X = activeMDS.update_embedding(S_TRAIN,X,len(S_TRAIN)*i,len(S_TRAIN)*(i+1))
    print "\n epoch = "+str((i+1))
    emp_loss,hinge_loss = activeMDS.get_loss(X,S_TRAIN)
    print "   TRAIN   emp_loss = "+str(emp_loss)+"   hinge_loss = "+str(hinge_loss)
    emp_loss,hinge_loss = activeMDS.get_loss(X,S_TEST)
    print "   TEST    emp_loss = "+str(emp_loss)+"   hinge_loss = "+str(hinge_loss)

