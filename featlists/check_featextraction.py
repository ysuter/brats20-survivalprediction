#!/usr/bin/env python3

import pandas as pd



# load training data features, extracted with parallel/batch method and sequentially
trainfeat_batch = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat.csv")
new_columns = trainfeat_batch.columns.values
new_columns[0] = 'BraTS20ID'
trainfeat_batch.columns = new_columns
trainfeat_batch = trainfeat_batch.set_index('BraTS20ID')

trainfeat_seq = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeat_NEW.csv")
new_columns = trainfeat_seq.columns.values
new_columns[0] = 'BraTS20ID'
trainfeat_seq.columns = new_columns
trainfeat_seq = trainfeat_seq.set_index('BraTS20ID')

# only keep relevant columns
trainfeat_batch = trainfeat_batch.loc[:, trainfeat_seq.columns.values]

# check if order of the rows is the same
print(trainfeat_batch.index.values == trainfeat_seq.index.values)

ne_stacked = (trainfeat_batch != trainfeat_seq).stack()
changed = ne_stacked[ne_stacked]
changed.index.names = ['id', 'col']

print(trainfeat_batch.equals(trainfeat_seq))
