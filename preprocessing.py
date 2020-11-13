import os
import requests
import json
from collections import Counter
import numpy as np
import pandas as pd
import itertools

ENDPOINT = 'https://api.census.gov/data/timeseries/eits/hv'
FIRST_YEAR, LAST_YEAR = 1965, 2020  # 1956 is first year with data, but doesn't have hor; 2020 exclusive
KEY = os.getenv('KEY')
COLUMNS = list(itertools.product(['HVR', 'RVR', 'HOR'], ['MW', 'NE', 'WE', 'SO', 'US']))

def make_year(year):
    PARAMS = {
        'get': ",".join(['data_type_code','cell_value','geo_level_code','time_slot_id']),
        'error_data': 'no',
        'seasonally_adj': 'no',
        'category_code': 'RATE',
        'time': year,
        'key': KEY
        }

    try:
        response = requests.get(ENDPOINT, params=PARAMS)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}; quitting...')
        quit()
    except Exception as err:
        print(f'Other error occurred: {err}; quitting...')
        quit()

    data = response.json()
    response_columns, data = data[0], data[1:]
    df_long = pd.DataFrame(data, columns=response_columns)[['time', 'data_type_code', 'geo_level_code', 'cell_value']]
    return df_long

def transform_year(year, df_long):
    wide_year = []
    for quarter in [f'{year}-Q{q}' for q in range(1,5)]:
        r = [quarter]
        for rate_type, geo_type in COLUMNS:
            filtered = df_long[(df_long['time'] == quarter) & (df_long['data_type_code'] == rate_type) & (df_long['geo_level_code'] == geo_type)] 
            assert(len(filtered) == 1)
            r.append(float(filtered['cell_value'].iloc[0]))
        wide_year.append(r)
    return wide_year

# print(data[0], data[1], len(data) - 1)


# at = lambda row, col_name: row[response_columns.index(col_name)]
# HVR = homeowner vacancy rate; RVR = rental vacancy rate; HOR = homeowner rate

# for col in response_columns:
    # if col != 'cell_value':
    # print(col, Counter(at(r, col) for r in data))

# print('zip', Counter(zip((at(r, 'data_type_code') for r in data), (at(r, 'category_code') for r in data))))

def make_wide():
    wide_columns = ['quarter']
    for rate_type, geo_type in COLUMNS:
        wide_columns.append(f'{rate_type}_{geo_type}')
    
    wide_data = []
    for year in range(FIRST_YEAR, LAST_YEAR):
        print(year)
        df_long = make_year(year)
        wide_data.extend(transform_year(year, df_long))

    # print(pd.DataFrame(wide_data, columns=wide_columns))

    wide_dict = {n: c for n, c in zip(wide_columns, zip(*wide_data))}
    # print(wide_dict)
    with open('data.json', 'w+') as f:
        json.dump(wide_dict, f)

make_wide()