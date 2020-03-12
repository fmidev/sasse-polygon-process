# -*- coding: utf-8 -*-
"""
CSV file handler
"""
import sys, logging, os, boto3
import numpy as np
#import datetime
#from configparser import ConfigParser
#import pandas.io.sql as sqlio
import pandas as pd
import geopandas as gpd
from shapely import wkt
from joblib import dump, load

class FileHandler(object):

    s3 = False
    gs = False
    bucket_name = ''
    bucket = ''
    client = ''

    """
    Create, store and read csv files
    """
    def __init__(self, s3_bucket=None):
        # self.config_filename = config_filename
        # self.config_name = config_name
        if s3_bucket is not None:
            self.bucket_name = s3_bucket
            self.s3 = True
            self.client = boto3.client('s3')
            resource = boto3.resource('s3')
            self.bucket = resource.Bucket(self.bucket_name)

    def dataset_from_csv(self, filename, time_column='point_in_time'):
        """
        Read dataset from csv file

        filename : str
                   filename to use

        return DataFrame
        """
        return pd.from_csv(filename, parse_dates=[time_column])

    def load_model(self, save_path, force=False):
        """ Load model from given path """
        filename = save_path + '/model.joblib'
        logging.info('Loading model from {}...'.format(filename))
        self._download_from_bucket(filename, filename, force=False)
        return load(filename)

    def save_model(self, model, save_path):
        """
        Save model to given path
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        filename = save_path + '/model.joblib'
        dump(model, filename)
        logging.info('Saved model to {}'.format(filename))

        if self.s3:
            self._upload_to_bucket(filename, filename)

    def df_to_csv(self, df, local_filename, ext_filename=None, store_header=True):
        """
        Store Pandas DataFrame to csv file and upload it to bucket if ext_filename is set
        """
        if df is not None and len(df) > 0:
            df.to_csv(local_filename, header=store_header, index_label='id')
            logging.info('Stored data to {}'.format(local_filename))
        else:
            open(local_filename, 'a').close()
            logging.info('No data, created empty file {}'.format(local_filename))

        if ext_filename is not None:
            self._upload_to_bucket(local_filename, ext_filename)

    def _upload_dir_to_bucket(self, path, ext_path):
        """
        Upload all files from folder to bucket
        """
        if self.s3:
            raise ValueError('S3 not implemented')
        if self.gs:
            for file in os.listdir(path):
                self._upload_to_bucket(path+'/'+file, ext_path+'/'+file)

    def _upload_to_bucket(self, filename, ext_filename):
        """
        Upload file to bucket if bucket is set and ext_filename is not None
        """
        if ext_filename is None:
            return

        if self.s3:
            self.bucket.upload_file(filename, ext_filename)
            logging.info('Uploaded {} to S3 with name {}'.format(filename, ext_filename))
        if self.gs:
            try:
                client = storage.Client()
                bucket = client.get_bucket(self.bucket_name)
                blob = storage.Blob(ext_filename, bucket)
                blob.upload_from_filename(filename)
                logging.info('Uploaded to {}'.format(ext_filename))
            except:
                logging.warning('Uploading file to bucket failed')

    def _download_dir_from_bucket(self, ext_path, local_path, force=False):
        """
        Download all files from bucket and save them to 'local_path'
        """
        if os.path.exists(local_path) and not force:
            logging.info('Path {} already exists. Not overwriting...'.format(local_path))
            return
        if os.path.exists(local_path) and force:
            logging.info('Path {} already exists. Overwriting...'.format(local_path))

        if self.s3:
            raise ValueError('S3 not implemented')
        if self.gs:
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=ext_path)

            for blob in blobs:
                local_name = blob.name.replace(ext_path, local_path)
                self._download_from_bucket(blob.name, local_name, force)

    def _download_from_bucket(self, ext_filename, local_filename, force=False):
        """
        Download file from bucket and save it to 'local_filename'
        """
        if os.path.exists(local_filename) and not force:
            logging.info('File {} already exists. Not overwriting...'.format(local_filename))
            return
        if os.path.exists(local_filename) and force:
            logging.info('File {} already exists. Overwriting...'.format(local_filename))

        if self.s3:
            self.bucket.download_file(ext_filename, local_filename)
            logging.info('Downloaded {} to {}'.format(ext_filename, local_filename))
        if self.gs:
            try:
                client = storage.Client()
                bucket = client.get_bucket(self.bucket_name)
                blob = storage.Blob(ext_filename, bucket)
                blob.download_to_filename(local_filename)
                logging.info('Downloaded {} to {}'.format(ext_filename, local_filename))
            except:
                logging.warning('Downloading failed')

                i += 1

    def write_csv(self, _dict, filename):
        """
        Write dict to csv

        _dict : dict
                data in format ['key': [values], 'key2': values]
        filename : str
                   filename where data is saved
        """
        with open(filename, 'w') as f:
            f.write('"'+'";"'.join(_dict.keys())+'"\n')
            for i in np.arange(len(_dict[list(_dict.keys())[0]])):
                values = []
                for col in _dict.keys():
                    try:
                        values.append(str(_dict[col][i]))
                    except IndexError as e:
                        # LSTM don't have first times available because of lacking history
                        pass
                f.write(';'.join(values)+'\n')

        logging.info('Wrote {}'.format(filename))
        self._upload_to_bucket(filename, filename)

    def report_cv_results(self, results, scores=['score'], filename=None, n_top=5):
        """
        Report CV results and save them to file
        """
        res = ""
        for score in scores:

            res += "{}\n".format(score)
            res += "-------------------------------\n"
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_{}'.format(score)] == i)
                for candidate in candidates:
                    res += "Model with rank: {0}\n".format(i)
                    res += "Mean validation {0}: {1:.3f} (std: {2:.3f})\n".format(
                        score,
                        results['mean_test_{}'.format(score)][candidate],
                        results['std_test_{}'.format(score)][candidate])
                    res += "Parameters: {0}\n".format(results['params'][candidate])
                    res += "\n"

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(res)

            self._upload_to_bucket(filename, filename)

        logging.info(res)
