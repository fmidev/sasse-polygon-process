import sys
sys.path.append("/usr/lib64/python2.7/site-packages")

import cv2
import numpy as np
import logging
from uuid import uuid4
import re

class StormObject(object):
    
    def __init__(self, polygon):
        self.polygon = polygon
        self.geom = None
        self.centroid = None
        self.bbox = None
        self.area = None
        self.identifier = str(uuid4())
        self.is_core_object = False
        self.is_visited = False
        self.belongs_to_cluster = False
        self.analysis_time = None
        self.enclosing_circle_radius = None
        self.enclosing_circle_center = None
        self.polygon_centroid()
        self.bounding_box()
        self.set_area()
        self.set_enclosing_circle()

    def set_area(self):
        self.area = cv2.contourArea(self.polygon)

    def polygon_centroid(self):
        """
        Calculates the centroid of a polygon and return None
        if centroid can not be calculated.
        """
        M = cv2.moments(self.polygon)
        try:    
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            return None
        self.centroid = (centroid_y, centroid_x)
        return (centroid_y, centroid_x)

    def get_contour_points_list(self):
        """
        Returns the polygon points of a storm object
        as a list [[x1,y1],[x2,y2],[x3,y3]...]
        """
        points = self.polygon.reshape((self.polygon.shape[0], 2)).tolist()
        return points

    def bounding_box(self):
        """
        Returns the min_y, max_y, min_x, max_x values for constructing the
        smallest straight rectangule around the stormobject.
        """
        x,y,w,h = cv2.boundingRect(self.polygon)            
        self.bbox = (y, y+h, x, x+w)

    def set_enclosing_circle(self):
        center, radius = cv2.minEnclosingCircle(self.polygon)
        self.enclosing_circle_center = np.array(center)
        self.enclosing_circle_radius = radius

    def geom_to_polygon(self, geo_trans):
        """ 
        Convert geometry to pixel polygons as a list
        [[x1,y1],[x2,y2],[x3,y3]...]
        """
        wkt = self.geom
        s = re.compile('(?<=\(\().*(?=\)\))')
        m = s.findall(wkt)
        coords = m[0].split(',')

        px_geom = []
        for coord in coords:
            lonlat = coord.split(' ')
            xy = self.world_to_pixel(geo_trans, float(lonlat[0]), float(lonlat[1]))
            px_geom.append(xy)

        self.polygon = np.array([px_geom])

        return px_geom

    def world_to_pixel(self, geoMatrix, x, y):
        """
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate
        """
        ulX = geoMatrix[0]
        ulY = geoMatrix[3]
        xDist = geoMatrix[1]
        yDist = geoMatrix[5]
        rtnX = geoMatrix[2]
        rtnY = geoMatrix[4]
        pixel = int((x - ulX) / xDist)
        line = int((ulY - y) / yDist)
        return [pixel, line]


        
