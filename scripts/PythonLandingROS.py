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
np.float = np.float64  # temp fix for ros_numpy
import ros_numpy
import std_msgs.msg
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest model

def train_random_forest(X_train, Y_train):
    rf = RandomForestRegressor(n_estimators=100)  # You can adjust the number of trees (n_estimators) as needed
    rf.fit(X_train, Y_train)
    return rf

def optimizer(ground_pcl, distance_to_top_leg, distance_to_right_leg, resolution, rf):
    # Convert ground point cloud to digital terrain model
    grid_size = [distance_to_right_leg/resolution[0], distance_to_top_leg/resolution[1]]
    x_min, x_max = np.min(ground_pcl[:, 0]), np.max(ground_pcl[:, 0])
    y_min, y_max = np.min(ground_pcl[:, 1]), np.max(ground_pcl[:, 1])
    xGrid = np.arange(x_min, x_max + grid_size[0], grid_size[0])
    yGrid = np.arange(y_min, y_max + grid_size[1], grid_size[1])
    XGrid, YGrid = np.meshgrid(xGrid, yGrid)

    # Predict terrain height using the trained random forest
    terrainModel = rf.predict(np.column_stack((XGrid.ravel(), YGrid.ravel()))).reshape(XGrid.shape)

    # Calculate points related to leg positions
    distance_to_top_leg_points = int(distance_to_top_leg/grid_size[1])
    distance_to_right_leg_points = int(distance_to_right_leg/grid_size[0])

    bottom_left_height = terrainModel[0:(len(yGrid) - distance_to_top_leg_points), 0:(len(xGrid) - distance_to_right_leg_points)]
    bottom_right_height = terrainModel[0:(len(yGrid) - distance_to_top_leg_points), (distance_to_right_leg_points + 1 - 1):(len(xGrid))]
    top_left_height = terrainModel[(distance_to_top_leg_points + 1 - 1):len(yGrid), 0:(len(xGrid) - distance_to_right_leg_points)]
    top_right_height = terrainModel[(distance_to_top_leg_points + 1 - 1):len(yGrid), (distance_to_right_leg_points + 1 - 1):(len(xGrid))]

    max_leg_height = np.fmax(np.fmax(np.fmax(bottom_left_height, bottom_right_height), top_left_height), top_right_height)
    min_leg_height = np.fmin(np.fmin(np.fmin(bottom_left_height, bottom_right_height), top_left_height), top_right_height)
    maximum_leg_height_difference = max_leg_height - min_leg_height

    # prepare plot of optimal landing locations
    fig, ax = plt.subplots()
    contourf_ = ax.contourf(XGrid[0:(len(yGrid) - distance_to_top_leg_points), 0:(len(xGrid) - distance_to_right_leg_points)], 
                YGrid[0:(len(yGrid) - distance_to_top_leg_points), 0:(len(xGrid) - distance_to_right_leg_points)], 
                maximum_leg_height_difference, 100, cmap='viridis', levels=np.linspace(0,0.2,100), extend = 'max')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Leg Differences (m)')
    plt.ylim()
    cbar = fig.colorbar(contourf_,ticks=[0,0.04,0.08,0.12,0.16,0.2])
    fig.canvas.draw()
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    # prepare plot of DEM
    fig, ax = plt.subplots()
    contourf_ = ax.contourf(XGrid[0:(len(yGrid)), 0:(len(xGrid))], 
                YGrid[0:(len(yGrid)), 0:(len(xGrid))], 
                terrainModel, 100, cmap='viridis')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Digital Elevation Model (m)')
    plt.ylim()
    fig.colorbar(contourf_)
    fig.canvas.draw()
    DEM_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    DEM_np = DEM_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return img_np, DEM_np


def convert_numpy_to_pc2_msg(numpy_pcl):
    fields = [sensor_msgs.msg.PointField('x', 0, sensor_msgs.msg.PointField.FLOAT32, 1),
              sensor_msgs.msg.PointField('y', 4, sensor_msgs.msg.PointField.FLOAT32, 1),
              sensor_msgs.msg.PointField('z', 8, sensor_msgs.msg.PointField.FLOAT32, 1)]
    header = std_msgs.msg.Header()
    header.frame_id = "map"
    header.stamp = rospy.Time.now()
    pcl_msg = sensor_msgs.point_cloud2.create_cloud(header,fields, numpy_pcl)
    return pcl_msg

def convert_numpy_to_img_msg(img_np):
    return ros_numpy.msgify(sensor_msgs.msg.Image, img_np, encoding = "rgb8")

def point_cloud_callback(point_cloud_msg, args):
    # unpack args
    pub_legs = args[0]
    pub_DEM = args[1]
    distance_to_top_leg = args[2]
    distance_to_right_leg = args[3]
    resolution = args[4]
    rf = args[5]  # Random Forest model

    # Convert ROS PointCloud2 message to Open3D PointCloud
    points_list = list(sensor_msgs.point_cloud2.read_points(point_cloud_msg, skip_nans=True, field_names = ("x", "y", "z")))
    points_list = np.asarray(points_list)

    if points_list.size > 3:
        # Run optimizer
        img_legs, img_DEM = optimizer(points_list, distance_to_top_leg, distance_to_right_leg, resolution, rf)

        # publish results of optimizer
        pub_legs.publish(convert_numpy_to_img_msg(img_legs))
        pub_DEM.publish(convert_numpy_to_img_msg(img_DEM))

def main():
    # Initialize the ROS node
    rospy.init_node('point_cloud_optimizer', anonymous=True)

    leg_position_rb = rospy.get_param('/leg_positions/rb')
    leg_position_lf = rospy.get_param('/leg_positions/lf')
    resolution = rospy.get_param('/terrain_model_resolution')

    distance_to_top_leg = abs(leg_position_rb[1] - leg_position_lf[1])
    distance_to_right_leg = abs(leg_position_rb[0] - leg_position_lf[0])

    # Load the trained Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)  # Modify parameters as needed
    # Assuming X_train, y_train are defined, replace them with your actual training data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])  # Example training data
    y_train = np.array([10, 20, 30])  # Example training labels
    rf.fit(X_train, y_train)

    # Create publishers for plot and DEM
    pub_legs = rospy.Publisher('/optimizer/optimal_landing_locations', sensor_msgs.msg.Image, queue_size = 1)
    pub_DEM = rospy.Publisher('/optimizer/digital_elevation_model', sensor_msgs.msg.Image, queue_size = 1)

    # Subscribe to the point cloud topic
    rospy.Subscriber('/zedm/zed_node/mapping/fused_cloud', sensor_msgs.msg.PointCloud2, point_cloud_callback, (pub_legs, pub_DEM, distance_to_top_leg, distance_to_right_leg, resolution, rf), queue_size=1)

    # Spin to keep the script from exiting
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
