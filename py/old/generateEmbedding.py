import os
import csv
from numpy import *
from numpy.random import *
import activeMDS
reload(activeMDS)

norm = linalg.norm
floor = math.floor
ceil = math.ceil

with open("/home/chris/PREP/animals/Animal Similarity Comparison-export-Thu Jun 26 16-27-40 CDT 2014.csv", "r") as f:
	headings = f.readline().strip().split(',')
	d = { k:[] for k in headings }
	cv = { k:[] for k in headings }
	q = headings.index('queryType')
	for line in f:
		data = line.strip().split(',')
		if int(data[q]) == 1:
			[d[k].append(v) for k,v in zip(headings,data)]
		else:
			[cv[k].append(v) for k,v in zip(headings,data)]

d['primary'] = [int(x) for x in d['primary']]
d['alternate'] = [int(x) for x in d['alternate']]
d['target'] = [int(x) for x in d['target']]

mm = min(d['primary']+d['alternate']+d['target'])
n = max(d['primary']+d['alternate']+d['target'])
S = [ [int(p)-mm,int(a)-mm,int(t)-mm] for p,a,t in zip(d['primary'],d['alternate'],d['target']) ]

X = randn(n,2)
X = X/norm(X)*sqrt(n)

X = activeMDS.update_embedding(S,X,0,len(S)*100)
