import sys
sys.path.insert(0, 'lib/')

import cv2
import numpy as np
import logging
from datetime import datetime
from datetime import timedelta
import hashlib, base64
import math

from shapely import wkt
from shapely.ops import nearest_points, transform
import pyproj
from functools import partial

import pandas as pd
import geopandas as gpd

from timepoint import TimePoint
from stormobject import StormObject
from dbhandler import DBHandler
from util import speed, bearing, get_storm_id


class Tracker(object):

    previous_timestamp = None
    speed_threshold = {'wind': 200, 'pressure': 45}
    distance_to_pressure_threshold = 500
    missing = -99

    def __init__(self, dbh, connective_overlap=.5, timestep=60):

        self.dbh = dbh
        self.optical_flows = []
        self.interpolated_time_points = []

        self.timestep = timestep # minutes
        self.connective_overlap = connective_overlap


    def optical_flow(self, frame1, frame2, method='farneback'):
        """
        Calculates dense optical flow field from two frames
        """
        #grayscale_frames = convert_to_grayscale([frame1, frame2])
        logging.debug("Start optical flow")
        if method == 'farneback':
            try:
                flow = cv2.calcOpticalFlowFarneback(frame1,
                                                    frame2,
                                                    None,
                                                    0.5,
                                                    3,
                                                    15,
                                                    3,
                                                    5,
                                                    1.2,
                                                    0)#OpenCV 3.0
            except:
                try:
                    flow = cv2.calcOpticalFlowFarneback(frame1,
                                                        frame2,
                                                        0.5,
                                                        3,
                                                        15,
                                                        3,
                                                        5,
                                                        1.2,
                                                        0) #OpenCV 2.4
                except Exception as e:
                    print(e)
        logging.debug("Finished calculating optical flow")
        return flow

    # def interpolate(self, frame, flow):
    #     """
    #     interpolate new frames using optical flow
    #     """
    #     h, w = flow.shape[:2]
    #     flow = -flow
    #     flow[:,:,0] += np.arange(w)
    #     flow[:,:,1] += np.arange(h)[:,np.newaxis]
    #     res = cv2.remap(frame, flow, None, cv2.INTER_NEAREST)
    #     return res

    def median_cluster_motion(self, optical_flow, cluster):
        cluster.create_cluster_mask(boolean_mask=True)
        cluster.bounding_box()
        cluster_optical_flow = optical_flow[cluster.bbox[0]:cluster.bbox[1]+1, cluster.bbox[2]: cluster.bbox[3]+1, :]
        points = cluster_optical_flow[cluster.cluster_mask.astype(bool)]
        median =  np.array([np.median(points[:][0]), np.median(points[:][1])]).astype(np.int32)
        return median

    def set_cluster_connectivity(self, previous_clusters, connective_overlap=0.5):
        """
        1. Interpolate clusters of previous time point to current time
        2. If interpolated cluster overlaps with current time point cluster the clusters are connected
        3. Add identifiers of connected previous time point clusters to current time point cluster
        """
        # Create an array with filled polygons from the previous time point
        # and use it to calculate optical flow
        logging.debug('set_cluster_connectivity start')
        logging.debug('Number of previous clusters {}'.format(len(previous_clusters)))
        logging.debug('Number of current clusters {}'.format(len(self.current_time_point.clusters)))

        filled_polygon_array = np.zeros_like(self.current_time_point.thresholded_array)
        for cluster in previous_clusters:
            cv2.fillPoly(filled_polygon_array,
                         [so.polygon for so in cluster.storm_objects],
                         255)

        optical_flow = self.optical_flow(filled_polygon_array,
                                         self.current_time_point.thresholded_array)
        # - create a boolean mask from each cluster of current time point
        # - slice the minimum rectangular area around the cluster from the mask
        # - add cluster, bounding box and the sliced mask to a list that is later later in the comparison
        logging.debug('create comparison arrays from current time point clusters')
        comparison_list = []
        for current_cluster in self.current_time_point.clusters:
            filled_polygon_array *= 0
            cv2.fillPoly(filled_polygon_array, [so.polygon for so in current_cluster.storm_objects], 255)
            current_cluster_mask = filled_polygon_array.astype(bool)
            current_cluster.bounding_box()
            comparison_mask = current_cluster_mask[current_cluster.bbox[0]:current_cluster.bbox[1]+1, current_cluster.bbox[2]:current_cluster.bbox[3]+1]

            bbox = current_cluster.bbox
            #bbox_center = np.array([int((b[0] + b[1])/2.), int((b[2] + b[3])/2.)])
            rect = (bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2])
            comparison_list.append((current_cluster, rect, comparison_mask))

        logging.debug('comparison arrays created, length {}'.format(len(comparison_list)))


        # Draw previous time point clusters on an array so that polygon points are moved
        # by median motion calculated from the optical flow of cluster area.
        # To compare the current time clusters to the previous ones, slices of current time cluster
        # bbox shape are taken from the above mentioned array.
        logging.debug('compare previous clusters to current time clusters')
        for previous_cluster in previous_clusters:
            filled_polygon_array *= 0

            median_motion = self.median_cluster_motion(optical_flow, previous_cluster)
            moved_previous_cluster = previous_cluster
            for so in moved_previous_cluster.storm_objects:
                so.polygon += median_motion

            cv2.fillPoly(filled_polygon_array, [so.polygon for so in moved_previous_cluster.storm_objects], 255)

            previous_cluster_mask  = filled_polygon_array.astype(bool)

            # Compare previous time point cluster mask to current time cluster masks.
            # If their bounding boxes intersect cluster overlap
            # percentage is calculated. If the overlap percentage is above the defined
            # limit, clusters are connected.
            moved_previous_cluster.bounding_box()
            bbox = moved_previous_cluster.bbox

            moved_cluster_rect = (bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2])
            for c in comparison_list:
                if rectangle_intersection(moved_cluster_rect, c[1]) is not None:
                    overlap_percentage = self.overlap_percentage(previous_cluster_mask[c[0].bbox[0]:c[0].bbox[1]+1, c[0].bbox[2]:c[0].bbox[3]+1], c[2])
                    # THIS PART IS VERY IMPORTANT
                    # if overlap_percentage threshold is to low
                    # false connections are created between clusters
                    # and it results in erroneous estimates in the
                    # nowcast phase
                    if overlap_percentage > connective_overlap:
                        c[0].connected_clusters.append(previous_cluster.identifier)

        logging.debug('cluster comparison finished')
        for cluster in self.current_time_point.clusters:
            logging.debug('cluster:{} conected: {}'.format(cluster.identifier, cluster.connected_clusters))


    def connect_polygons(self, polygons_1, polygons_2):
        """
        Connect polygons and calculate trakcing parameters (speed, angle, area, area_diff)
        """
        # If current polygons are empty, return empy
        if len(polygons_1) < 1:
            return polygons_1

        # If previous polygons are empty, all storms are new. Return a new storm id and empty tracking fields
        # TODO should we assign storm id here or when we have it at least on two time steps?
        if len(polygons_2) < 1:
            polygons_1['storm_id'], polygons_1['speed'], polygons_1['angle'], \
            polygons_1['area_m2'], polygons_1['area_diff'], \
            polygons_1['speed_pressure'], polygons_1['angle_pressure'], \
            polygons_1['distance_to_pressure'] = None, self.missing, self.missing, self.missing, self.missing, self.missing, self.missing, self.missing
            return polygons_1

        # Add centroid if necessary
        if 'centroid' not in polygons_1.columns:
            polygons_1.loc[:, 'centroid'] =  polygons_1.loc[:,'geom'].centroid

        if 'centroid' not in polygons_2.columns:
            polygons_2.loc[:, 'centroid'] =  polygons_2.loc[:,'geom'].centroid

        # Separate wind and pressure elements
        polygons_1_wind = polygons_1[(polygons_1.loc[:,'weather_parameter'] != 'Pressure')]
        polygons_2_wind = polygons_2[(polygons_2.loc[:,'weather_parameter'] != 'Pressure')]
        polygons_1_pressure = polygons_1[(polygons_1.loc[:,'weather_parameter'] == 'Pressure')]
        polygons_2_pressure = polygons_2[(polygons_2.loc[:,'weather_parameter'] == 'Pressure')]

        centroids_pressure_1 = gpd.GeoSeries(polygons_1_pressure.loc[:, 'centroid']).unary_union
        centroids_pressure_2 = gpd.GeoSeries(polygons_2_pressure.loc[:, 'centroid']).unary_union

        def near(needle_row, search_base):

            # Nearest pressure objects
            try:
                nearest_pressure_1 = polygons_1_pressure[(polygons_1_pressure.centroid == nearest_points(needle_row.centroid, centroids_pressure_1)[1])].iloc[0]
                # Distance to pressure object
                distance_to_pressure = speed(needle_row, nearest_pressure_1)
                # Use only pressure polygons which are near enough
                if distance_to_pressure > self.distance_to_pressure_threshold:
                    nearest_pressure_1 = []

            except ValueError:
                distance_to_pressure = self.missing
                nearest_pressure_1 = []

            try:
                # Nearest previous object
                centroids = gpd.GeoSeries(search_base[(search_base.loc[:, 'weather_parameter'] == needle_row.weather_parameter) &
                                                      (search_base.loc[:, 'low_limit'] == needle_row.low_limit) &
                                                      (search_base.loc[:, 'high_limit'] == needle_row.high_limit)].loc[:,'centroid']).unary_union

                nearest = search_base[(search_base.centroid == nearest_points(needle_row.centroid, centroids)[1])].iloc[0]
                nearest_pressure_2 = polygons_2_pressure[(polygons_2_pressure.centroid == nearest_points(nearest.centroid, centroids_pressure_2)[1])].iloc[0]

                # Speed and bearing of the object and nearest pressure objects
                speed_self = speed(needle_row, nearest)
                angle_self = bearing(nearest.centroid, needle_row.centroid)

                if len(nearest_pressure_1) >= 1 and len(nearest_pressure_2) >= 1:
                    speed_pressure = speed(nearest_pressure_1, nearest_pressure_2, self.speed_threshold['pressure'], self.missing)
                    angle_pressure = bearing(nearest_pressure_2.centroid, nearest_pressure_1.centroid)
                else:
                    speed_pressure, angle_pressure = self.missing, self.missing

                separate = False
                if speed_self > self.speed_threshold['wind'] or speed_pressure > self.speed_threshold['pressure']:
                    logging.warning('Nearest object over {} km away. Treating as a separate storm'.format(self.speed_threshold))
                    separate = True
                    speed_self = self.missing
            except ValueError:
                # Happens when no similar polygons exist in search_base and centroids becomes empty
                separate = True
                speed_self, angle_self = self.missing, self.missing
                speed_pressure, angle_pressure = self.missing, self.missing
                distance_to_pressure = self.missing

            # area
            g = transform(
                partial(
                    pyproj.transform,
                    pyproj.Proj(init='EPSG:4326'),
                    pyproj.Proj(
                        proj='aea',
                        lat_1=needle_row.geom.bounds[1],
                        lat_2=needle_row.geom.bounds[3])),
                needle_row.geom)
            area_m2 = g.area

            # Assign storm id. If set in previous polygon, use that
            if not separate and nearest.storm_id is not None:
                storm_id = nearest.storm_id
                area_diff = area_m2 - nearest.area_m2
            else:
                storm_id = get_storm_id(needle_row)
                area_diff = area_m2

            # Assign to pressure polgyons as well
            if len(nearest_pressure_1) >= 1:
                old_id = polygons_1[(polygons_1.loc[:, 'id'] == nearest_pressure_1.id)].iloc[0].storm_id
                polygons_1.loc[polygons_1.loc[:, 'id'] == nearest_pressure_1.id, 'storm_id'] = storm_id
                if old_id is not None and old_id != storm_id:
                    logging.warning('Changing storm-id for polygon {} ({} -> {})'.format(nearest_pressure_1.id, old_id, storm_id))

            return storm_id, speed_self, angle_self, area_m2, area_diff, speed_pressure, angle_pressure, distance_to_pressure

        # Apply to all rows in the dataframe
        polygons_1['storm_id'], polygons_1['speed'], polygons_1['angle'], \
        polygons_1['area_m2'], polygons_1['area_diff'], polygons_1['speed_pressure'], \
        polygons_1['angle_pressure'], polygons_1['distance_to_pressure'] = zip(*polygons_1.apply(lambda row: near(row, polygons_2), axis=1))

        print(polygons_1)
        return polygons_1


    def overlap_percentage(self, mask1, mask2):
        combined_mask = mask1 * mask2
        n_overlapping_elements = sum(combined_mask.flatten())
        overlap = n_overlapping_elements/float(sum(mask2.flatten()))
        return overlap

    def run(self, timestamp):
        """
        Run tracker

        1. load storm objects for current time point from db
        2. load storm object data of previous time point from db
        3. match the storm objects of previous time point to storm objects of current time point
        4. set cluster connectivity
        5. store to db

        timestamp : DateTime
                    timestamp to process

        """

        # 1. load storm objects for current time point from db
        logging.debug("Loading data from db for time: {}...".format(timestamp))
        storm_objects = self.dbh.get_polygons({'time': [timestamp.strftime('%Y-%m-%d %H:%M:%S')]})
        logging.debug('Found {} storm objects'.format(len(storm_objects)))

        # 2. load storm object data of previous time point from db
        # We assume that data comes at fixed time steps
        previous_timestamp = timestamp - timedelta(minutes=self.timestep)
        if self.previous_timestamp == previous_timestamp:
            logging.debug('Using storm objects from memory...')
            previous_storm_objects = self.previous_storm_objects
        else:
            logging.debug("Reading previous objects from db for time: {}...".format(previous_timestamp))
            previous_storm_objects = self.dbh.get_polygons({'time': [previous_timestamp.strftime('%Y-%m-%d %H:%M:%S')]})
            logging.debug('Found {} storm objects'.format(len(previous_storm_objects)))

        # print(storm_objects)
        # sys.exit()
        #df = self.form_df(storm_objects)
        storm_objects = self.connect_polygons(storm_objects, previous_storm_objects)

        # Keep previous storm objects in memory so we don't have to load them again
        self.previous_storm_objects = storm_objects
        self.previous_timestamp = timestamp

        return
        sys.exit()
        # # 4. identify clusters for current time point
        # cluster_neighbourhood_limit = 2 #km
        # cluster_min_area = 20 #km^2
        # pixel_width = 0.25 # km  TODO: extract pixel width from geotiff
        # neighbourhood_radius = cluster_neighbourhood_limit / float(pixel_width)
        # area_limit = cluster_min_area / float(pixel_width**2)
        # logging.debug("Do the DBSCAN")
        # self.current_time_point.dbscan(
        #     neighbourhood_radius=neighbourhood_radius, area_limit=area_limit)

        # 4. set cluster connectivity
        logging.debug("Setting cluster connectivity...")
        previous_clusters = self.database.read_clusters(previous_timestamp)
        if len(previous_clusters) > 0:
            self.set_cluster_connectivity(previous_clusters, connective_overlap=self.connective_overlap)

        # # 5. Insert clusters into db
        # logging.debug("Insert clusters to db")
        # self.database.insert_clusters(self.current_time_point.clusters,
        #                               self.current_time_point)

        # 5. Store to db
        logging.debug("Insert current objects to db")
        self.database.insert_storm_objects(self.current_time_point)
        logging.debug("Done, exiting.")
