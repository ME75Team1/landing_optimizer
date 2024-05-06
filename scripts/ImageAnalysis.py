import rospy
import numpy as np
import open3d as o3d
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import ros_numpy
numpy.float = float
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg

class TerrainAnalyzer:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image, self.depth_callback)
        self.color_sub = rospy.Subscriber('/zedm/zed_node/rgb/image_rect_color', Image, self.color_callback)

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg)
        # Perform terrain analysis using depth image

        # Example: Calculate terrain elevation statistics
        min_elevation = np.min(depth_image)
        max_elevation = np.max(depth_image)
        mean_elevation = np.mean(depth_image)
        median_elevation = np.median(depth_image)

        print("Min Elevation:", min_elevation)
        print("Max Elevation:", max_elevation)
        print("Mean Elevation:", mean_elevation)
        print("Median Elevation:", median_elevation)

        # Example: Terrain classification based on elevation
        classified_terrain = self.classify_terrain(depth_image)

    def color_callback(self, msg):
        color_image = self.bridge.imgmsg_to_cv2(msg)
        # Perform terrain analysis using color image (optional)

        # Example: Obstacle detection using color image
        obstacle_mask = self.detect_obstacles(color_image)

    def classify_terrain(self, depth_image):
        # Placeholder for terrain classification algorithm
        # Example: Simple thresholding based on elevation
        threshold_value = 0.5  # Adjust threshold value as needed
        classified_terrain = np.where(depth_image > threshold_value, 1, 0)  # Binary classification

        return classified_terrain

    def detect_obstacles(self, color_image):
        # Placeholder for obstacle detection algorithm
        # Example: Color-based segmentation for obstacle detection
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 100, 100])  # Lower bound of obstacle color in HSV space
        upper_bound = np.array([10, 255, 255])  # Upper bound of obstacle color in HSV space
        obstacle_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)  # Binary mask of obstacles

        return obstacle_mask

def convert_numpy_to_pc2_msg(numpy_pcl):
    fields = [std_msgs.msg.PointField('x', 0, std_msgs.msg.PointField.FLOAT32, 1),
              std_msgs.msg.PointField('y', 4, std_msgs.msg.PointField.FLOAT32, 1),
              std_msgs.msg.PointField('z', 8, std_msgs.msg.PointField.FLOAT32, 1)]
    header = std_msgs.msg.Header()
    header.frame_id = "map"
    header.stamp = rospy.Time.now()
    pcl_msg = pc2.create_cloud(header, fields, numpy_pcl)
    return pcl_msg

def point_cloud_callback(point_cloud_msg, terrain_analyzer):
    # Convert ROS PointCloud2 message to numpy array
    point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(point_cloud_msg)

    # Perform terrain analysis
    terrain_analyzer.analyze_terrain(point_cloud)

def main():
    rospy.init_node('terrain_analyzer', anonymous=True)
    terrain_analyzer = TerrainAnalyzer()

    # Create a subscriber to the point cloud topic
    rospy.Subscriber('/zedm/zed_node/mapping/fused_cloud', PointCloud2, point_cloud_callback, terrain_analyzer, queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    main()
