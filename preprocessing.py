import os
import json
import csv
import zipfile

superpath = './WESAD'
big = []

types_to_extract = ['HR']

for s in os.listdir(superpath):
    subpath = os.path.join(superpath, s)
    if os.path.isdir(subpath):
        print(s)
        sample = []
        zipfilepath, folderpath = f'{s}_E4_Data.zip', f'{s}_E4_Data'
        datafolder = os.path.join(subpath, folderpath)
        if folderpath not in os.listdir(subpath):
            datasubpath = os.path.join(subpath, zipfilepath)
            with zipfile.ZipFile(datasubpath, 'r') as zip_ref:
                zip_ref.extractall(datafolder)
        for datatype in types_to_extract:
            with open(os.path.join(datafolder, f'{datatype}.csv')) as f:
                sample.append([float(val.replace('\n', '')) for val in f.readlines()[2:]])
    big.append(sample)

with open('data_smaller.json', 'w') as f:
    json.dump(big, f)