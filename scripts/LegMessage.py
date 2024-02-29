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
    leg_positions = np.array([[0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]])  
    z_values = points[:, 2]  # Extract z values from the point cloud
    interpolator = NearestNeighborInterpolator(points[:, :2], z_values)

    leg_heights = []
    for pos in leg_positions:
        z = interpolator(pos[0], pos[1])
        leg_heights.append(z)

    return leg_positions, leg_heights

def point_cloud_callback(point_cloud_msg, arg):
    leg_positions, leg_heights = process_point_cloud(point_cloud_msg)
    publf = arg[0]
    publb = arg[1]
    pubrf = arg[2]
    pubrb = arg[3]
    # Create and publish a message for each leg position
    for i, height in enumerate(leg_heights):
        point_msg = std_msgs.msg.Float64()
        point_msg = height
        if i == 0:
            pubrf.publish(point_msg)
        elif i == 1:
            publf.publish(point_msg)
        elif i == 2:
            pubrb.publish(point_msg)
        elif i == 3:
            publb.publish(point_msg)

def main():
    rospy.init_node('leg_position_publisher', anonymous=True)

    publf = rospy.Publisher('leg_positions/lf', std_msgs.msg.Float64, queue_size=10)
    publb = rospy.Publisher('leg_positions/lb', std_msgs.msg.Float64, queue_size=10)
    pubrf = rospy.Publisher('leg_positions/rf', std_msgs.msg.Float64, queue_size=10)
    pubrb = rospy.Publisher('leg_positions/rb', std_msgs.msg.Float64, queue_size=10)

    rospy.Subscriber('/zed2/zed_node/point_cloud/cloud_registered', PointCloud2, point_cloud_callback, (publf, publb, pubrf, pubrb), queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
