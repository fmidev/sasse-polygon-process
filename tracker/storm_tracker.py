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

def save_dataset(tracker, d, fh=None, db=None):
    """
    Save dataset to db or csv file depending on given command line options
    """
    if options.dataset_type == 'csv':
        filename = '{}_sasse_2_dataset.csv'.format(d.strftime('%Y%m%d'))
        fh.df_to_csv(tracker.dataset, filename, filename)

    if options.dataset_type == 'db':
        db.update_classification_dataset(tracker.dataset)

    tracker.dataset = None

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
    ymd = '{}{}{}'.format(d.year, d.month, d.day)
    while d <= endtime:
        logging.info('Processing time: {}'.format(d))
        tracker.run(d)
        d += timedelta(hours=1)
        if ymd != '{}{}{}'.format(d.year, d.month, d.day):
            save_dataset(tracker, d, fh=fh, db=dbh)
        ymd = '{}{}{}'.format(d.year, d.month, d.day)

    #if tracker.dataset is not None and len(tracker.dataset) > 0:
    save_dataset(tracker, d, fh=fh, db=dbh)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2011-02-01T00:00:00', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2011-03-01T00:00:00', help='End time of the classification data interval')
    parser.add_argument('--connective_overlap', type=float, default=0.5, help='minimum overlap between connected clusters')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='production', help='Section name in db cnf file to read connection parameters')
    parser.add_argument('--smartmet_config_filename', type=str, default='cnf/smartmet.yaml', help='CNF file containing SmartMet Server pararemters')
    parser.add_argument('--smartmet_config_name', type=str, default='production', help='Section name for smartmet')
    parser.add_argument('--dataset_type', type=str, default='db', help='Store dataset to (db|csv)')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.DEBUG)

    main()
