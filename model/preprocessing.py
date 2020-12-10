import os
import json
import csv
import zipfile

superpath = './WESAD'

def one_feature():
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


def multiple_features():
    big = []
    types_to_extract = ['HR', 'BVP']

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
            
            hr_len = None
            for datatype in types_to_extract:
                with open(os.path.join(datafolder, f'{datatype}.csv')) as f:
                    raw_data = [float(val.replace('\n', '')) for val in f.readlines()]
                if datatype == 'HR': 
                    sample.append(raw_data[2:])
                    hr_len = len(raw_data[2:])
                if datatype == 'BVP': 
                    rate = int(raw_data[1])
                    raw_data = raw_data[2+10:]  # starts 10 sec before hr data
                    means = []
                    for start_ind in range(0, len(raw_data), rate):
                        if start_ind + rate > len(raw_data): 
                            continue
                        end_ind = start_ind + rate
                        means.append(sum(raw_data[start_ind:end_ind]) / rate)
                    sample.append(means[:hr_len])

        sample_T = list(zip(*sample))
        if sample_T not in big: big.append(sample_T)

    with open('data_multiple_features.json', 'w') as f:
        json.dump(big, f)

