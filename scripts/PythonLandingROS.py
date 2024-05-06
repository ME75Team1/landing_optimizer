import rospy
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import scipy.stats
import sensor_msgs
import sensor_msgs.point_cloud2
import sensor_msgs.msg
from scipy.spatial import cKDTree
from pykrige.ok import OrdinaryKriging  # Import OrdinaryKriging from pykrige.ok module
np.float = np.float64  
import ros_numpy
import std_msgs.msg

def convert_numpy_to_img_msg(img_np):
    return ros_numpy.msgify(sensor_msgs.msg.Image, img_np, encoding="rgb8")

def point_cloud_callback(point_cloud_msg, args):
    pub_legs, pub_DEM, distance_to_top_leg, distance_to_right_leg, resolution = args

    points_list = list(sensor_msgs.point_cloud2.read_points(point_cloud_msg, skip_nans=True, field_names=("x", "y", "z")))
    points_list = np.asarray(points_list)

    if points_list.size >= 3:  # Ensure at least three points are available for optimization
        img_legs, img_DEM = optimizer(points_list, distance_to_top_leg, distance_to_right_leg, resolution)
        pub_legs.publish(convert_numpy_to_img_msg(img_legs))
        pub_DEM.publish(convert_numpy_to_img_msg(img_DEM))

def main():
    rospy.init_node('point_cloud_optimizer', anonymous=True)

    leg_position_rb = rospy.get_param('/leg_positions/rb')
    leg_position_lf = rospy.get_param('/leg_positions/lf')
    resolution = rospy.get_param('/terrain_model_resolution')

    distance_to_top_leg = abs(leg_position_rb[1] - leg_position_lf[1])
    distance_to_right_leg = abs(leg_position_rb[0] - leg_position_lf[0])

    pub_legs = rospy.Publisher('/optimizer/optimal_landing_locations', sensor_msgs.msg.Image, queue_size=1)
    pub_DEM = rospy.Publisher('/optimizer/digital_elevation_model', sensor_msgs.msg.Image, queue_size=1)

    rospy.Subscriber('/zedm/zed_node/mapping/fused_cloud', sensor_msgs.msg.PointCloud2, point_cloud_callback,
                     (pub_legs, pub_DEM, distance_to_top_leg, distance_to_right_leg, resolution), queue_size=1)

    rospy.spin()

def optimizer(ground_pcl, distance_to_top_leg, distance_to_right_leg, resolution):
    grid_size = [distance_to_right_leg/resolution[0], distance_to_top_leg/resolution[1]]
    x_min, x_max = np.min(ground_pcl[:, 0]), np.max(ground_pcl[:, 0])
    y_min, y_max = np.min(ground_pcl[:, 1]), np.max(ground_pcl[:, 1])
    xGrid = np.arange(x_min, x_max + grid_size[0], grid_size[0])
    yGrid = np.arange(y_min, y_max + grid_size[1], grid_size[1])
    XGrid, YGrid = np.meshgrid(xGrid, yGrid)

    OK = OrdinaryKriging(ground_pcl[:, 0], ground_pcl[:, 1], ground_pcl[:, 2], variogram_model='linear',
                         enable_plotting=False)  # disable plotting to avoid potential issues
    terrainModel, ss = OK.execute('grid', XGrid, YGrid)

    # Create images directly from the terrain model data
    img_legs = terrainModel.copy()  # Assuming terrainModel represents leg differences
    img_DEM = terrainModel.copy()   # Assuming terrainModel represents digital elevation model

    return img_legs, img_DEM

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
