import rospy
from sensor_msgs.msg import Imu
import math

class ZEDMiniOrientationNode:
    def __init__(self):
        rospy.init_node('zed_mini_orientation_node', anonymous=True)

        # Set up subscriber to ZED Mini IMU topic
        rospy.Subscriber('/zed/zed_node/imu/data', Imu, self.imu_callback)

    def imu_callback(self, msg):
        # Extract orientation from IMU message
        orientation = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)

        # Print orientation
        print("Current Orientation (degrees):")
        print("Roll: {:.2f}".format(math.degrees(roll)))
        print("Pitch: {:.2f}".format(math.degrees(pitch)))
        print("Yaw: {:.2f}".format(math.degrees(yaw)))

    def quaternion_to_euler(self, x, y, z, w):
        # Convert quaternion to Euler angles
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    zed_mini_orientation_node = ZEDMiniOrientationNode()
    zed_mini_orientation_node.run()
