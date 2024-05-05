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
from pykrige.ok import OrdinaryKriging  
np.float = np.float64  
import ros_numpy
import std_msgs.msg

from pykrige.ok import OrdinaryKriging
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import scipy.interpolate

class KrigingTree(object):
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)

    def fit(self, X=None, z=None, leafsize=10):
        return self.__init__(X, z, leafsize)

    def __call__(self, X, k=3, eps=1e-6, p=2, regularize_by=1e-9):
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        OK = OrdinaryKriging(X[:, 0], X[:, 1], weights, variogram_model='linear')
        z, ss = OK.execute('grid', X[:, 0], X[:, 1])
        return z

    def transform(self, X, k=3, p=2, eps=1e-6, regularize_by=1e-9):
        return self.__call__(X, k, eps, p, regularize_by)

def optimizer(ground_pcl, distance_to_top_leg, distance_to_right_leg, resolution):
    grid_size = [distance_to_right_leg/resolution[0], distance_to_top_leg/resolution[1]]
    x_min, x_max = np.min(ground_pcl[:, 0]), np.max(ground_pcl[:, 0])
    y_min, y_max = np.min(ground_pcl[:, 1]), np.max(ground_pcl[:, 1])
    xGrid = np.arange(x_min, x_max + grid_size[0], grid_size[0])
    yGrid = np.arange(y_min, y_max + grid_size[1], grid_size[1])
    XGrid, YGrid = np.meshgrid(xGrid, yGrid)
    OK = OrdinaryKriging(ground_pcl[:, 0], ground_pcl[:, 1], ground_pcl[:, 2], variogram_model='linear')
    terrainModel, ss = OK.execute('grid', XGrid, YGrid)

    distance_to_top_leg_points = int(distance_to_top_leg/grid_size[1])
    distance_to_right_leg_points = int(distance_to_right_leg/grid_size[0])

    bottom_left_height = terrainModel[0:(len(yGrid) - distance_to_top_leg_points), 0:(len(xGrid) - distance_to_right_leg_points)]
    bottom_right_height = terrainModel[0:(len(yGrid) - distance_to_top_leg_points), (distance_to_right_leg_points + 1 - 1):(len(xGrid))]
    top_left_height = terrainModel[(distance_to_top_leg_points + 1 - 1):len(yGrid), 0:(len(xGrid) - distance_to_right_leg_points)]
    top_right_height = terrainModel[(distance_to_top_leg_points + 1 - 1):len(yGrid), (distance_to_right_leg_points + 1 - 1):(len(xGrid))]

    max_leg_height = np.fmax(np.fmax(np.fmax(bottom_left_height, bottom_right_height), top_left_height), top_right_height)
    min_leg_height = np.fmin(np.fmin(np.fmin(bottom_left_height, bottom_right_height), top_left_height), top_right_height)
    maximum_leg_height_difference = max_leg_height - min_leg_height

    # Plot optimal landing locations
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

    # Plot DEM
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
    pcl_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, numpy_pcl)
    return pcl_msg

def convert_numpy_to_img_msg(img_np):
    return ros_numpy.msgify(sensor_msgs.msg.Image, img_np, encoding="rgb8")

def point_cloud_callback(point_cloud_msg, args):
    pub_legs = args[0]
    pub_DEM = args[1]
    distance_to_top_leg = args[2]
    distance_to_right_leg = args[3]
    resolution = args[4]

    points_list = list(sensor_msgs.point_cloud2.read_points(point_cloud_msg, skip_nans=True, field_names=("x", "y", "z")))
    points_list = np.asarray(points_list)

    if points_list.size > 3:
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

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
