import cv2
import numpy as np
from stormobject import StormObject
#import osgeo

class TimePoint(object):

    # def __init__(self, data_array):
    #     """
    #     Container for the data of one time point.
    #     Parameters
    #     ----------
    #     data_array : nympy array
    #     time_index : int
    #     """
    #     self.data_array = data_array.GetRasterBand(1).ReadAsArray()
    #     self.dataset = data_array
    #     self.thresholded_array = None
    #     self.contoured_array = None
    #     self.contours = None
    #     self.storm_objects = []
    #     self.clusters = []
    #     self.timestamp = None
    def __init__(self, df):
        """
        Container for the data of one time point.
        Parameters
        ----------
        df : pandas DataFrame
        time_index : int
        """
        #self.data_array = data_array.GetRasterBand(1).ReadAsArray()
        self.data_array = df.loc[:, 'geom']
        self.dataset = df
        self.thresholded_array = None
        self.contoured_array = None
        self.contours = None
        self.storm_objects = []
        self.clusters = []
        self.timestamp = None
        self.identify_storm_objects()


    def test_polygon_distances(self):
        """test method used only in debugging
        """
        distance_dict = {}
        for so1 in self.storm_objects:
            distance_dict[str(so1.identifier)] = []
            for so2 in self.storm_objects:
                val = (so2.identifier, self.polygon_border_distance(so1.polygon, so2.polygon))
                distance_dict[str(so1.identifier)].append(val)
        return distance_dict

    def test_polygon_neighborhood_areas(self, radius):
        """test method used only in debugging
        """
        neighborhood_areas_dict = {}
        for so in self.storm_objects:
            nearby_storm_objects = self.nearby_storm_objects(so, radius)
            area_sum = sum([nso.area for nso in nearby_storm_objects])
            neighborhood_areas_dict[str(so.identifier)] = area_sum
        return neighborhood_areas_dict

    # def threshold_array(self, dbz_threshold, noise_threshold):
    #     """Creates a binary thresholded array.
    #     """
    #     img = self.data_array[:]
    #     img[img == 255] = 0
    #
    #     offset = -32 # 8-bit
    #     gain = 0.5 # 8-bit
    #     threshold = (dbz_threshold - offset)/gain
    #     img[img < (noise_threshold-offset)/gain] = 0
    #     ret, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    #
    #     # Morphological closing, structuring element is 3 km x 3 km, pixel width is 0.25 km
    #     #close_kernel = np.ones((6,6),np.uint8)
    #     open_kernel = np.ones((3,3),np.uint8)
    #     close_kernel = np.ones((12,12),np.uint8)
    #
    #     thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, close_kernel)
    #     thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, open_kernel)
    #     self.thresholded_array = thresholded_img

    def identify_contours(self):
        """Gets the contours from thresholded array
        """
        try:
            contour_img, contours, hierarchy = cv2.findContours(self.thresholded_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #OpenCV 3.0
            self.contours = contours
            self.contoured_array = contour_img
        except:
            try:
                contours, hierarchy = cv2.findContours(self.thresholded_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # OpenCV 2.4
                self.contours = contours
            except Exception as e:
                print(e)

    # def identify_storm_objects(self, dbz_threshold=35, noise_threshold=8):
    #     """
    #     Identifies the storm objects by thresholding the image and finding
    #     contours from it.
    #     """
    #     self.threshold_array(dbz_threshold, noise_threshold)
    #     self.identify_contours()
    #     for contour in self.contours:
    #         self.storm_objects.append(StormObject(contour))

    def identify_storm_objects(self, dbz_threshold=35, noise_threshold=8):
        """
        Identifies the storm objects by thresholding the image and finding
        contours from it.
        """
        for polygon in self.data_array:
            self.storm_objects.append(StormObject(polygon))        

    # OLD, SLOW AND MORE GENERAL METHOD, NOT IN USE
    def polygon_border_distance(self, polygon1, polygon2):
        """
        Returns the minimum distance between two polygon borders
        """
        if len(polygon1) < len(polygon2):
            polygon = polygon2
            points = [tuple(*p) for p in polygon1.tolist()]
        else:
            polygon = polygon1
            points = [tuple(*p) for p in polygon2.tolist()]

        border_distances = [cv2.pointPolygonTest(polygon, point, True) for point in points]
        # Validate distances:
        # Distances are negative if point is outside the polygon,
        # positive if inside and zero if at the border
        distance_signs = [True for x in border_distances if x < 0]
        if all(distance_signs):
            return abs(max(border_distances))
        else:
            return 0

    # OLD VERSION OF THE METHOD
    def nearby_storm_objects_old(self, storm_object, neighborhood_radius):
        """
        Returns a list of storm_objects whose minimum border
        distance is smaller than neighborhood_radius
        """
        nearby_storm_objects = []
        for so in self.storm_objects:
            distance = self.polygon_border_distance(storm_object.polygon, so.polygon)
            if distance < neighborhood_radius:
                nearby_storm_objects.append(so)

        return nearby_storm_objects


    def nearby_storm_objects(self, storm_object, neighborhood_radius):
        """
        Returns a list of storm_objects whose border
        distance is smaller than neighborhood_radius
        """
        nearby_storm_objects = []
        for so in self.storm_objects:
            # Rough comparison:
            # if the border distance of two polygon enclosing circles
            # is smaller than the neighborhood radius, a more detailed calculation is made
            center_distance = so.enclosing_circle_center - storm_object.enclosing_circle_center
            boundary_distance = np.hypot(*center_distance) - \
                                so.enclosing_circle_radius - storm_object.enclosing_circle_radius
            if boundary_distance <= neighborhood_radius:
                if len(so.polygon) < len(storm_object.polygon):
                    polygon = storm_object.polygon
                    points = [tuple(*p) for p in so.polygon.tolist()]
                else:
                    polygon = so.polygon
                    points = [tuple(*p) for p in storm_object.polygon.tolist()]

                for point in points:
                    if cv2.pointPolygonTest(polygon, point, True) < neighborhood_radius:
                        nearby_storm_objects.append(so)
                        break
        return nearby_storm_objects

    def is_core_object(self, storm_objects, area_limit):
        """
        Returns True if the sum of nearby storm_object areas is greater than
        area_limit
        """
        area_sum = sum([so.area for so in storm_objects])
        if area_sum > area_limit:
            return True
        else:
            return False

    def dbscan(self, neighbourhood_radius=2, area_limit=80):
        """
        DBSCAN clusterning algorithm for identifying storm cell clusters.

        Algorithm description:
        1. Loop trough all storm objects until an unvisited core object is found.
        2. Create a new cluster and call the recursive function expand_cluster
        giving the corepoint and the neighboring points as argument.

        """
        for so in self.storm_objects:
            if so.is_visited:
                continue
            so.is_visited = True

            nearby_storm_objects = self.nearby_storm_objects(so, neighbourhood_radius)
            if self.is_core_object(nearby_storm_objects, area_limit):
                cluster = Cluster()
                self.expand_cluster(so, nearby_storm_objects, cluster, neighbourhood_radius, area_limit )
                self.clusters.append(cluster)

    def expand_cluster(self, storm_object, nearby_storm_objects, cluster, neighbourhood_radius, area_limit):
        """
        Add storm object to cluster, called in dbscan
        """
        cluster.storm_objects.append(storm_object)
        storm_object.belongs_to_cluster = True

        for so in nearby_storm_objects:
            if not so.is_visited:
                so.is_visited = True
                new_nearby_storm_objects = self.nearby_storm_objects(so, neighbourhood_radius)
                if self.is_core_object(new_nearby_storm_objects, area_limit):
                    self.expand_cluster(so, new_nearby_storm_objects, cluster, neighbourhood_radius, area_limit)
            else:
                if not so.belongs_to_cluster:
                    cluster.storm_objects.append(so)
                    so.belongs_to_cluster = True
