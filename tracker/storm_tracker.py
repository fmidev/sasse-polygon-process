# -*- coding: utf-8 -*-
import sys
import argparse
import logging
from tracker import Tracker
from dbhandler import DBHandler
from filehandler import FileHandler
from smartmethandler import SmartMetHandler
import datetime as dt
from datetime import timedelta

def main():

    # Read in data and extract data arrays
    #logging.info("Reading input files.")

    dbh = DBHandler(options.db_config_filename, options.db_config_name)
    ssh = SmartMetHandler(options.smartmet_config_filename, options.smartmet_config_name)
    fh = FileHandler(s3_bucket='fmi-sasse-classification-dataset')
    tracker = Tracker(dbh, ssh)

    starttime = dt.datetime.strptime(options.starttime, "%Y-%m-%dT%H:%M:%S")
    endtime = dt.datetime.strptime(options.endtime, "%Y-%m-%dT%H:%M:%S")

    d = starttime
    ym = '{}{}'.format(d.year, d.month)
    while d <= endtime:
        logging.info('Processing time: {}'.format(d))
        tracker.run(d)
        if ym != '{}{}'.format(d.year, d.month):
            filename = '{}_sasse_2_dataset.csv'.format(ym)
            fh.df_to_csv(tracker.dataset, filename, filename)
        d += timedelta(hours=1)
        ym = '{}{}'.format(d.year, d.month)

    #if tracker.dataset is not None and len(tracker.dataset) > 0:
    filename = '{}_sasse_2_dataset.csv'.format(ym)
    fh.df_to_csv(tracker.dataset, filename, filename)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2011-02-01T00:00:00', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2011-03-01T00:00:00', help='End time of the classification data interval')
    parser.add_argument('--connective_overlap', type=float, default=0.5, help='minimum overlap between connected clusters')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='production', help='Section name in db cnf file to read connection parameters')
    parser.add_argument('--smartmet_config_filename', type=str, default='cnf/smartmet.yaml', help='CNF file containing SmartMet Server pararemters')
    parser.add_argument('--smartmet_config_name', type=str, default='production', help='Section name for smartmet')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    main()
