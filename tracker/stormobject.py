import sys
# sys.path.append("/usr/lib64/python2.7/site-packages")

# import cv2
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
        # self.bounding_box()
        # self.set_area()
        # self.set_enclosing_circle()

    def set_area(self):
        self.area = cv2.contourArea(self.polygon)

    def polygon_centroid(self):
        """
        Calculates the centroid of a polygon and return None
        if centroid can not be calculated.
        """
        self.centroid = self.polygon.centroid
    
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
