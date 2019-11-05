#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial import KDTree
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
IMAGE_CLASSIFICATION_THRESHOLD = 3
USE_TRAFFIC_LIGHT_CLASSIFIER = 1 #if using classifier = 1, otherwise 0
PUBLISH_TL_WITHOUT_CAMERA = 0 #if not using camera for TL = 1, otherwise 0 for normal operation

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        rospy.logwarn('Initializing TL_Detector - 001')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.waypoints_2d = None
        self.waypoints_2d_init = False
        self.waypoints_tree = None
        self.image_count = 0
        self.last_time = rospy.get_time()
        self.sample_time = 0.
        
        rospy.logwarn('Initializing TL_Detector - 002')

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        rospy.logwarn('Initializing TL_Detector - 003')

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        rospy.logwarn('Initializing TL_Detector - 004')


        rospy.spin()

    def pose_cb(self, msg):
        #rospy.logwarn("pose_cb")
        self.pose = msg
        if (PUBLISH_TL_WITHOUT_CAMERA == 1) and (not self.waypoints_tree):
            light_wp, state = self.process_traffic_lights()
            #rospy.logwarn("Traffic light state {}" .format(state))
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def waypoints_cb(self, waypoints):
        #rospy.logwarn("waypoints_cb")
        rospy.logwarn('Initializing TL_Detector - 005')
        self.waypoints = waypoints
        if not self.waypoints_2d_init:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)
            self.waypoints_2d_init = True

    def traffic_cb(self, msg):
        #rospy.logwarn("traffic_cb")
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        curr_time = rospy.get_time()
        self.sample_time += (curr_time - self.last_time)
        self.last_time = curr_time
        if self.sample_time > 1:
            self.sample_time = 0
            return # as we just started getting camera image
        elif self.sample_time < 0.3:
            return
        #rospy.logwarn("sample time is {:f}" .format(self.sample_time))
        self.sample_time = 0.

        

        
        #rospy.logwarn("image_cb")
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if state is not None:
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #rospy.logwarn("get_closest_waypoint")

        closest_idx = self.waypoints_tree.query([x,y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if USE_TRAFFIC_LIGHT_CLASSIFIER == 0:
            #rospy.logwarn("Traffic light color is: {}" .format(light.state))
            return light.state # for testing purpose only (once classifier remove this line)
        else:
            if(not self.has_image):
                self.prev_light_loc = None
                return False

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            #Get classification
            self.image_count += 1
            if(self.image_count <= IMAGE_CLASSIFICATION_THRESHOLD):
                return
            self.image_count = 0 # reset the count
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #rospy.logwarn("process_traffic_lights")
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                #rospy.logwarn("Info: {} {}" .format(d, diff))
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
