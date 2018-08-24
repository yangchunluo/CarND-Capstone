#!/usr/bin/env python
"""
Traffic light detector module.
"""
from cv_bridge import CvBridge
import rospy
from scipy.spatial import KDTree
import tf
import yaml

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane

from light_classification.tl_classifier import TLClassifier

_STATE_COUNT_THRESHOLD = 3
_SPIN_FREQUENCY = 30


class TLDetector(object):
    """
    Traffic light detector node.
    """
    def __init__(self, enable_classification=True):
        rospy.init_node('tl_detector')
        self.enable_classification = enable_classification

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_list = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_callback)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_callback)
        rospy.Subscriber('/image_color', Image, self.image_callback)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.config["is_site"])
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

    def spin(self, freq):
        """
        Spins this ROS node based on the given frequency.

        :param freq: frequency in hertz.
        """
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():

            # Publish upcoming red lights at camera frequency.
            # Each predicted state has to occur `_STATE_COUNT_THRESHOLD` number
            # of times till we start using it. Otherwise the previous stable state is used.
            if None not in (self.pose, self.waypoints, self.camera_image):
                light_wp, state = self.process_traffic_lights()
                # Once traffic light is processed, set camera_image to None.
                self.camera_image = None

                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= _STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1
            rate.sleep()

    def pose_callback(self, msg):
        self.pose = msg

    def waypoints_callback(self, waypoints):
        self.waypoints = waypoints

        # Get the waypoints in X, Y plane and set up the KDTree for efficient comparison.
        self.waypoints_2d = [[w.pose.pose.position.x, w.pose.pose.position.y]
                             for w in waypoints.waypoints]
        self.waypoints_tree = KDTree(self.waypoints_2d)

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        for i, stop_line_position in enumerate(stop_line_positions):
            closest_idx = self.waypoints_tree.query([stop_line_position[0], stop_line_position[1]], 1)[1]
            self.stopline_list.append(closest_idx)

    def traffic_callback(self, msg):
        self.lights = msg.lights

    def image_callback(self, msg):
        self.camera_image = msg

    def get_closest_waypoint(self, pose):
        """
        Gets the closest path waypoint to the given position.
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem

        Args:
            pose (Pose): position to match a waypoint to.
        Returns:
            int: index of the closest waypoint in self.waypoints.
        """
        x = pose.position.x
        y = pose.position.y
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        return closest_idx

    def process_traffic_lights(self):
        """
        Finds closest visible traffic light, if one exists, and determines its location and color.

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        assert self.pose
        car_wp = self.get_closest_waypoint(self.pose.pose)
        for ix, stop_wp in enumerate(self.stopline_list):
            if stop_wp < car_wp:
                continue
            if self.enable_classification:
                assert self.camera_image
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                state = self.light_classifier.get_classification(cv_image)
            else:
                state = self.lights[ix]
            return stop_wp, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector().spin(_SPIN_FREQUENCY)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
