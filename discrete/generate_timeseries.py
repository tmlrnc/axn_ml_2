import argparse

from discrete.vl_kmeans_kmedian import K_Means, normalizer
from discrete.binize import VL_Binizer
from discrete.binize_kmeans import VL_Discretizer_KMeans
import csv

import pandas
import pandas as pd
import numpy
import numpy as np


from ohe.encoder import OneHotEncoderBuilder
from discrete.discretizer import DiscretizerBuilder


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_ohe')
    parser.add_argument('--file_out_discrete')
    parser.add_argument('--file_out_ohe_dis')

    parser.add_argument(
        '--drop',
        action='append')
    parser.add_argument('--file_out')
    parser.add_argument('--dicretize', nargs='+',  action='append')



    args = parser.parse_args()
    return args

def main():
    """
  READ FILE_IN_RAW.CSV
  GET COLUMN HEADERS
  FOR EACH COLUMN NOT IN IGNORE LIST :
  GET ALL CATEGORIES = UNIQUE COLUMN VALUES
  GENERATE ONE HOT ENCODING HEADER
  ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER

      """
    ######################################################################
    #
    # read run commands
    #
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_discrete = args.file_out_discrete
    file_out = args.file_out

    file_out_ohe_dis = args.file_out_ohe_dis
    vl_dicretize_list_many = args.dicretize

    ######################################################################
    #
    # Discretize
    #
    print("Discretize --- START ")
    file_in_name_org = file_in_name
    file_out_org = file_out



    #import random
    #for i in range(10000):
    #    print(str(random.random()) + ",")
    #exit()

    df = pandas.read_csv(file_in_name_org).fillna(value=0)

    import csv
    filename = 'covid_joe7.csv'
    outfile = 'covid_joe7_pivot.csv'
    headers = []
    headerset = set()
    regions = {}
    with open(filename) as io:
        reader = csv.DictReader(io)
        for r in reader:
            print(r)
            region = r['Country']
            date = r['\ufeffDate']
            if region not in regions:
                regions[region] = {}
            regions[region][date] = {'confirmed': r['confirmed'],
                                     'deaths': r['deaths'],
                                     'recovered': r['recovered']
                                     }
    records = []
    for region in regions.keys():
        r = {}
        r['Country'] = region
        for date, data in regions[region].items():
            confirmed = date + '_confirmed'
            deaths = date + '_deaths'
            recovered = date + '_recovered'
            r[confirmed] = data['confirmed']
            r[deaths] = data['deaths']
            r[recovered] = data['recovered']
            for h in [confirmed, deaths, recovered]:
                if h in headerset:
                    continue
                headers.append(h)
                headerset.add(h)
        records.append(r)
    headers.sort(reverse=True)
    headers = ['Country'] + headers
    with open(outfile, 'w') as io:
        writer = csv.DictWriter(io, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    print("Discretize --- END ")



if __name__ == '__main__':
    main()


