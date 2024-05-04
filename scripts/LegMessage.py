import rospy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator
import sensor_msgs.msg
from sensor_msgs.msg import PointCloud2
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from motor_controller.msg import legHeights


class NearestNeighborInterpolator:
    def __init__(self, points, values):
        self.interpolator = NearestNDInterpolator(points, values)

    def __call__(self, x, y):
        return self.interpolator(x, y)

def process_point_cloud(point_cloud_msg):
    # Convert ROS PointCloud2 message to Open3D PointCloud
    points_list = list(pc2.read_points(point_cloud_msg, skip_nans=True, field_names=("x", "y", "z")))
    points = np.asarray(points_list)
    ptCloud = o3d.geometry.PointCloud()
    ptCloud.points = o3d.utility.Vector3dVector(points)

    # Assuming the legs are defined by specific x, y positions
    lf_position = rospy.get_param("/leg_positions/lf")
    rf_position = rospy.get_param("/leg_positions/rf")
    lb_position = rospy.get_param("/leg_positions/lb")
    rb_position = rospy.get_param("/leg_positions/rb")
    leg_positions = np.array([rf_position, lf_position, rb_position, lb_position]).astype(float)
    depth_values = points[:, 0]  # Extract z values from the point cloud
    interpolator = NearestNeighborInterpolator(points[:, 1:], depth_values)

    leg_heights = legHeights()
    leg_heights.ids = [rospy.get_param('/dynamixel_ids/lf'), rospy.get_param('/dynamixel_ids/rf'),rospy.get_param('/dynamixel_ids/rb'),rospy.get_param('/dynamixel_ids/lb')]
    leg_heights.heights = [interpolator(lf_position[0], lf_position[1]), interpolator(rf_position[0], rf_position[1]), interpolator(rb_position[0], rb_position[1]), interpolator(lb_position[0], lb_position[1])]
    return leg_heights

def point_cloud_callback(point_cloud_msg, pub):
    leg_heights = process_point_cloud(point_cloud_msg)
    print(leg_heights)
    pub.publish(leg_heights)

def convert_string_to_list(list_string):
    list_list = list_string.strip('][').split(', ')
    return(list_list)

def main():
    rospy.init_node('leg_position_publisher', anonymous=True)

    pub = rospy.Publisher('/leg_heights', legHeights, queue_size=1)

    rospy.Subscriber('/zedm/zed_node/point_cloud/cloud_registered', PointCloud2, point_cloud_callback, (pub), queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
