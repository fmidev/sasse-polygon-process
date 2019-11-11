# -*- coding: utf-8 -*-
import sys
import argparse
import logging
from tracker import Tracker
from dbhandler import DBHandler
import datetime as dt
from datetime import timedelta

def main():

    # Read in data and extract data arrays
    #logging.info("Reading input files.")

    dbh = DBHandler(options.db_config_filename, options.db_config_name)
    tracker = Tracker(dbh)

    starttime = dt.datetime.strptime(options.starttime, "%Y-%m-%d")
    endtime = dt.datetime.strptime(options.endtime, "%Y-%m-%d")

    d = starttime
    while d <= endtime:
        tracker.run(d)
        d += timedelta(hours=1)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2011-02-01', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2011-03-01', help='End time of the classification data interval')
    parser.add_argument('--connective_overlap', type=float, default=0.5, help='minimum overlap between connected clusters')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='local', help='Section name in db cnf file to read connection parameters')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.DEBUG)

    main()
