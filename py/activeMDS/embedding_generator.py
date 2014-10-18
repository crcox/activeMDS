#!/usr/bin/env python

import activeMDS as mds
import csv
import json
import os
import sys

class InputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

jsonfile = sys.argv[1]

with open(jsonfile,'rb') as f:
    master = json.load(f)

for special in master['configs']:
    config = master
    for k,v in special.items():
        config[k] = v

    name = config['name']

    if master['condor']:
        outdir = os.path.join(config['condor']['staging'],config['name'])
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        with open(os.path.join(outdir,'config.json'),'wb') as f:
            json.dump(config, f, sort_keys=True, indent=2, separators=(',', ': '))

    elif master['local']:
        outdir = os.path.join(config['local']['output'],config['name'])
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        responses = mds.read_triplets(master['data'])
        model = mds.initializeEmbedding(responses['nitems'],master['ndim'])
        lossLog = mds.fitModel(model, responses, config)

        with open(os.path.join(outdir,'loss.csv'),'wb') as f:
            writer = csv.writer(f)
            writer.writerows(lossLog)

        with open(os.path.join(outdir,'embedding.csv'),'wb') as f:
            writer = csv.writer(f)
            writer.writerows(model)

        with open(os.path.join(outdir,'config.json'),'wb') as f:
            json.dump(config, f, sort_keys=True, indent=2, separators=(',', ': '))
    else:
        raise InputError('Neither condor or local settings specified.')
