import rospy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator
import sensor_msgs.msg
from sensor_msgs.msg import PointCloud2
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import ros_numpy

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

def point_cloud_callback(point_cloud_msg, pub):
    leg_positions, leg_heights = process_point_cloud(point_cloud_msg)

    # Create and publish a message for each leg position
    for i, (pos, height) in enumerate(zip(leg_positions, leg_heights)):
        point_msg = geometry_msgs.msg.Point()
        point_msg.x = pos[0]
        point_msg.y = pos[1]
        point_msg.z = height
        pub.publish(point_msg)

def main():
    rospy.init_node('leg_position_publisher', anonymous=True)

    pub = rospy.Publisher('leg_positions', geometry_msgs.msg.Point, queue_size=10)

    rospy.Subscriber('/zed2/zed_node/mapping/fused_cloud', PointCloud2, point_cloud_callback, pub, queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
