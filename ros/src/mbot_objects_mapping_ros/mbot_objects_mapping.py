#! /usr/bin/env python
from math import *
import numpy as np

import rospy
from mbot_perception_msgs.msg import *
from mbot_perception_msgs.srv import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from copy import deepcopy


def euclidean_distance(x, y, z):
    return sqrt(x ** 2 + y ** 2 + z ** 2)


def l2_norm(v2, v1=None):
    if v1 is None:
        return euclidean_distance(v2.x, v2.y, v2.z)
    else:
        return euclidean_distance(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z)


class MappedObject:

    def __init__(self, obj):
        self.tracked_object = obj
        self.mapped_object = MappedObject3D()


class ObjectMap(dict):

    def __init__(self):
        super(ObjectMap, self).__init__()
        self.header = None
        self.changed = False

    def __setitem__(self, key, tracked_obj):

        if key not in self:
            # Insert the object if still not in the map
            super(ObjectMap, self).__setitem__(key, MappedObject(tracked_obj))
        else:
            self[key].tracked_object = tracked_obj

        # Update mapped_object's values from tracked_object
        max_i = np.argmax(self[key].tracked_object.class_probability)
        self[key].mapped_object.uuid = self[key].tracked_object.uuid
        self[key].mapped_object.class_name = self[key].tracked_object.class_name[max_i]
        self[key].mapped_object.confidence = self[key].tracked_object.class_probability[max_i]
        self[key].mapped_object.pose = self[key].tracked_object.pose

    def __str__(self):
        s = ""
        for obj in self.values():
            s += "%s:\n" % obj.tracked_object.uuid
            max_i = np.argmax(obj.tracked_object.class_probability)
            for i in range(len(obj.tracked_object.class_name)):
                s += "%15s  %s  %1.3f\n" % (
                    obj.tracked_object.class_name[i], '*' if i == max_i else ' ',
                    obj.tracked_object.class_probability[i])
        return s

    def get_by_class_name(self, class_name):
        list_by_class_name = []

        for obj in self.values():
            if obj.mapped_object.class_name == class_name:
                list_by_class_name.append(obj)

        return list_by_class_name

    def to_msg(self):

        mapped_objects = MappedObject3DList(header=self.header)

        # Iterate through all objects and find the most likely class for each
        for obj in self.values():
            mapped_objects.objects.append(obj.mapped_object)

        return mapped_objects


class ObjectsMapping(object):

    def __init__(self):

        # variables
        self.map = ObjectMap()
        self.objects_count = rospy.get_param("~objects_count")
        self.max_covariance = rospy.get_param("~max_covariance", 0.1)
        self.marker_array_publisher = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

        # subscribers
        tracked_objects_topic = rospy.get_param("~tracked_objects", "")
        self.tracked_objects_subscriber = rospy.Subscriber(tracked_objects_topic, TrackedObject3DList,
                                                           self.tracked_objects_callback, queue_size=10)

        # publishers
        mapped_objects_topic = rospy.get_param("~mapped_objects", "")
        self.mapped_objects_publisher = rospy.Publisher(mapped_objects_topic, MappedObject3DList, queue_size=10)

        # service servers
        rospy.Service('~get_map', GetObjectsMap3D, self.get_objects_map)
        rospy.Service('~delete_object', DeleteObject3D, self.delete_object)

    def valid_object(self, obj):
        """
        :type obj: TrackedObject3D
        """

        cov = np.matrix(obj.pose.covariance)
        cov = cov.reshape(6, 6)

        return euclidean_distance(cov[0, 0], cov[2, 2], cov[4, 4]) < self.max_covariance

    def tracked_objects_callback(self, msg):
        self.map.header = msg.header
        for obj in msg.objects:
            uuid = obj.uuid
            if self.valid_object(obj):
                self.map[uuid] = obj
                self.map.changed = True

    def get_objects_map(self, _):
        return GetObjectsMap3DResponse(self.map.to_msg())

    def delete_object(self, req):
        if req.uuid in self.map:
            del self.map[req.uuid]
            rospy.loginfo("Deleted object from map, uuid: %s" % req.uuid)
        else:
            rospy.logwarn("Requested to delete object from map, but object was not in map, uuid: %s" % req.uuid)
        return DeleteObject3DResponse()

    def publish_map(self):
        self.mapped_objects_publisher.publish(self.map.to_msg())
        self.map.changed = False
        rospy.loginfo("publish map: \n%s" % self.map)

    def loop(self):
        sleeper = rospy.Rate(50)
        prev_max_i = 0

        while not rospy.is_shutdown():

            if self.map.changed:
                self.publish_map()

                marker_array = MarkerArray()

                # Renumber the marker IDs
                i = 0
                for obj in self.map.values():

                    if obj.mapped_object is not None:
                        text_marker = Marker()
                        text_marker.header.frame_id = self.map.header.frame_id
                        text_marker.type = Marker.TEXT_VIEW_FACING
                        text_marker.ns = 'mapped_objects'
                        text_marker.id = i
                        text_marker.text = obj.mapped_object.class_name
                        text_marker.scale.z = 0.05
                        text_marker.color = ColorRGBA(1, 1, 1, 1)
                        text_marker.pose = deepcopy(obj.mapped_object.pose.pose)
                        text_marker.pose.position.z += 0.1
                        i += 1

                        marker_array.markers.append(text_marker)

                        marker = Marker()
                        marker.header.frame_id = self.map.header.frame_id
                        marker.type = Marker.SPHERE
                        marker.ns = 'mapped_objects'
                        marker.id = i
                        marker.scale = Vector3(0.05, 0.05, 0.05)
                        marker.color = ColorRGBA(1, 0, 0, 1)
                        marker.pose.orientation.w = 1.0
                        marker.pose = obj.mapped_object.pose.pose
                        i += 1

                        marker_array.markers.append(marker)

                for delete_i in range(i, prev_max_i):
                    marker_array.markers.append(Marker(id=delete_i, ns='mapped_objects', action=Marker.DELETE))

                prev_max_i = i

                # Publish the MarkerArray
                self.marker_array_publisher.publish(marker_array)

            sleeper.sleep()
