import numpy as np
import pandas as pd
import sklearn.cluster as cl

# read in files
items = ["RawData",
         "IAV-M_NEG_RawData",
         "IAV-M_POS_RawData",
         "IBV-M_NEG_RawData",
         "IBV-M_POS_RawData",
         "MHV_NEG_RawData",
         "MHV_POS_RawData",
         "RSV-N_NEG_RawData",
         "RSV-N_POS_RawData",
         "SARS-N1_NEG_RawData",
         "SARS-N1_POS_RawData",
         "SARS-N2_NEG_RawData",
         "SARS-N2_POS_RawData",
        ]
path_map = {}
df_all_map = {}
for name in items:
    path_map[name] = f'../Data/6P-wastewater-samples-labelled/droplet-level-data/RawData/6P-wastewater-samples-labelled_S-0697825-9_A3_{name}.csv'
    df_all_map[name] = pd.read_csv(path_map[name])

# make one big df
#print(df_all_map.keys())
df_all = df_all_map["RawData"]
print(df_all.columns)
# spalten [x,y, 1,2,3,4,5,6, index]
np_features = df_all.drop(["x-coordinate_in_pixel"," y-coordinate_in_pixel"," index"],axis=1)
np_features = np_features.to_numpy()

classifier = cl.KMeans(13)
out = classifier.fit_transform(np_features)
print(out)
