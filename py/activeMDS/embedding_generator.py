import activeMDS as mds
import csv
import json
import sys

jsonfile = sys.argv[1]

with open(jsonfile,'rb') as f:
    master = json.load(f)

responses = mds.read_triplets(master['data'])

for special in master['configs']:
    config = master
    for k,v in special.items():
        config[k] = v
    model = mds.initializeEmbedding(responses['nitems'],master['ndim'])
    lossLog = mds.fitModel(model, responses, config)
    with open('data/config1/loss.csv','wb') as f:
        writer = csv.writer(f)
        writer.writerows(lossLog)

    with open('data/config1/embedding.csv','wb') as f:
        writer = csv.writer(f)
        writer.writerows(model)
