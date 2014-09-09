import activeMDS as mds
import csv
import json
import os
import sys

jsonfile = sys.argv[1]

with open(jsonfile,'rb') as f:
    master = json.load(f)

responses = mds.read_triplets(master['data'])

if not os.path.isdir(os.path.join('embeddings')):
    os.makedirs(os.path.join('embeddings'))


for special in master['configs']:
    config = master
    for k,v in special.items():
        config[k] = v

    name = config['name']

    outdir = os.path.join('embeddings',config['name'])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir,'config.json'),'wb') as f:
        json.dump(config, f)

    if master['condor'] == False:
        model = mds.initializeEmbedding(responses['nitems'],master['ndim'])
        lossLog = mds.fitModel(model, responses, config)

        with open(os.path.join(outdir,'loss.csv'),'wb') as f:
            writer = csv.writer(f)
            writer.writerows(lossLog)

        with open(os.path.join(outdir,'embedding.csv'),'wb') as f:
            writer = csv.writer(f)
            writer.writerows(model)
