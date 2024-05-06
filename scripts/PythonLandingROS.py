import rospy
import numpy as np
import open3d as o3d
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sensor_msgs
from scipy.spatial import cKDTree
import ros_numpy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import sensor_msgs.point_cloud2
import sensor_msgs.msg
float = np.float64  #mport rospy

import std_msgs.msg

class RBFInterpolator(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.

    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.

    Returns:
    --------
        RBFInterpolator instance: object

    Example:
    --------

    # 'train'
    rbf_interpolator = RBFInterpolator(X1, z1)

    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    X2 = np.reshape(X2, (2, -1)).T
    z2 = rbf_interpolator(X2)

    """

    def __init__(self, X=None, z=None):
        if not X is None:
            self.X = X
            self.z = z
            self.weights = None

    def fit(self, X=None, z=None):
        """
        Initialize the RBF interpolation model.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N sample points in a d-dimensional space.
            z: (N,) ndarray
                Corresponding scores.

        Returns:
        --------
            RBFInterpolator instance: object
        """
        self.X = X
        self.z = z
        self.weights = None

    def __call__(self, X):
        """
        Perform RBF interpolation.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        if self.weights is None:
            self.weights = self._compute_weights()

        distances = self._compute_distances(X)
        z = np.dot(distances, self.weights)
        return z

    def _compute_weights(self):
        """
        Compute the weights for RBF interpolation.

        Returns:
        --------
            weights: (N,) ndarray
                Interpolation weights.
        """
        distances = self._compute_distances(self.X)
        weights = np.linalg.lstsq(distances, self.z, rcond=None)[0]
        return weights

    def _compute_distances(self, X):
        """
        Compute the distances between query points and sample points.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

        Returns:
        --------
            distances: (N, M) ndarray
                Distances between query points and sample points.
        """
        N = X.shape[0]
        M = self.X.shape[0]
        distances = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                distances[i, j] = np.linalg.norm(X[i] - self.X[j])
        return distances

def optimizer(ground_pcl, distance_to_top_leg, distance_to_right_leg, resolution):
    # Convert ground point cloud to digital terrain model
    grid_size = [distance_to_right_leg/resolution[0], distance_to_top_leg/resolution[1]]
    x_min, x_max = np.min(ground_pcl[:, 0]), np.max(ground_pcl[:, 0])
    y_min, y_max = np.min(ground_pcl[:, 1]), np.max(ground_pcl[:, 1])
    xGrid = np.arange(x_min, x_max + grid_size[0], grid_size[0])
    yGrid = np.arange(y_min, y_max + grid_size[1], grid_size[1])
    XGrid, YGrid = np.meshgrid(xGrid, yGrid)
    rbf_interpolator = RBFInterpolator(ground_pcl[:, 0:2], ground_pcl[:,2])
    X2 = np.meshgrid(xGrid, yGrid)
    grid_shape = X2[0].shape
    X2 = np.reshape(X2, (2, -1)).T
    z2 = rbf_interpolator(X2)
    terrainModel = z2.reshape(grid_shape)

    # distance_to_top_leg = 0.08
    # distance_to_right_leg = 0.06
    distance_to_top_leg_points = int(distance_to_top_leg/grid_size[1])
    distance_to_right_leg_points = int(distance_to_right_leg/grid_size[0])

    # get locations of each leg as a function of the position of bottom left
    # leg
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

    return(img_np, DEM_np)

def convert_numpy_to_pc2_msg(numpy_pcl):
    fields = [sensor_msgs.msg.PointField('x', 0, sensor_msgs.msg.PointField.FLOAT32, 1),
              sensor_msgs.msg.PointField('y', 4, sensor_msgs.msg.PointField.FLOAT32, 1),
              sensor_msgs.msg.PointField('z', 8, sensor_msgs.msg.PointField.FLOAT32, 1)]
    header = std_msgs.msg.Header()
    header.frame_id = "map"
    header.stamp = rospy.Time.now()
    pcl_msg = sensor_msgs.point_cloud2.create_cloud(header,fields, numpy_pcl)
    return(pcl_msg)

def convert_numpy_to_img_msg(img_np):
    return(ros_numpy.msgify(sensor_msgs.msg.Image, img_np, encoding = "rgb8"))

def point_cloud_callback(point_cloud_msg, args):
    # unpack args
    pub_legs = args[0]
    # pub_ground = args[1]
    pub_DEM = args[1]
    distance_to_top_leg = args[2]
    distance_to_right_leg = args[3]
    resolution = args[4]

    # Convert ROS PointCloud2 message to Open3D PointCloud
    points_list = list(sensor_msgs.point_cloud2.read_points(point_cloud_msg, skip_nans=True, field_names = ("x", "y", "z")))
    points_list = np.asarray(points_list)

    if points_list.size > 3:
        # Run optimizer
        img_legs, img_DEM = optimizer(points_list, distance_to_top_leg, distance_to_right_leg, resolution)

        # publish results of optimizer
        pub_legs.publish(convert_numpy_to_img_msg(img_legs))
        pub_DEM.publish(convert_numpy_to_img_msg(img_DEM))
        # pub_ground.publish(convert_numpy_to_pc2_msg(ground_pcl))
        # pub_nonground.publish(convert_numpy_to_pc2_msg(nonground_pcl))

def main():
    # Initialize the ROS node
    rospy.init_node('point_cloud_optimizer', anonymous=True)

    leg_position_rb = rospy.get_param('/leg_positions/rb')
    leg_position_lf = rospy.get_param('/leg_positions/lf')
    resolution = rospy.get_param('/terrain_model_resolution')

    distance_to_top_leg = abs(leg_position_rb[1] - leg_position_lf[1])
    distance_to_right_leg = abs(leg_position_rb[0] - leg_position_lf[0])

    print(distance_to_top_leg, distance_to_right_leg)
    # Create a publisher to publish plot
    pub_legs = rospy.Publisher('/optimizer/optimal_landing_locations', sensor_msgs.msg.Image, queue_size = 1)
    # pub_ground = rospy.Publisher('/optimizer/ground_point_cloud', sensor_msgs.msg.PointCloud2, queue_size = 1)
    # pub_nonground = rospy.Publisher('/optimizer/non_ground_point_cloud', sensor_msgs.msg.PointCloud2, queue_size = 1)
    pub_DEM = rospy.Publisher('/optimizer/digital_elevation_model', sensor_msgs.msg.Image, queue_size = 1)

    # Create a subscriber to the point cloud topic
    # Replace 'input_point_cloud_topic' with your actual topic name
    rospy.Subscriber('/zedm/zed_node/mapping/fused_cloud', sensor_msgs.msg.PointCloud2, point_cloud_callback, (pub_legs, pub_DEM, distance_to_top_leg, distance_to_right_leg, resolution), queue_size=1)

    # Spin to keep the script for exiting
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
