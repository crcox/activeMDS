import json
import activeMDS as mds

model = mds.initializeEmbedding(responses['nitems'],master['ndim'])
lossLog = mds.fitModel(model, responses, config)
with open(os.path.join(outdir,'loss.csv'),'wb') as f:
    writer = csv.writer(f)
    writer.writerows(lossLog)

with open(os.path.join(outdir,'embedding.csv'),'wb') as f:
    writer = csv.writer(f)
    writer.writerows(model)
