import struct
import threading
import time
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped
from Go2Py.joy import xKeySwitch, xRockerBtn
from geometry_msgs.msg import TwistStamped
from unitree_go.msg import LowState, Go2pyLowCmd
from nav_msgs.msg import Odometry   
from scipy.spatial.transform import Rotation
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R



def ros2_init(args=None):
    rclpy.init(args=args)


def ros2_close():
    rclpy.shutdown()

class ROS2ExecutorManager:
    """A class to manage the ROS2 executor. It allows to add nodes and start the executor in a separate thread."""
    def __init__(self):
        self.executor = MultiThreadedExecutor()
        self.nodes = []
        self.executor_thread = None

    def add_node(self, node: Node):
        """Add a new node to the executor."""
        self.nodes.append(node)
        self.executor.add_node(node)

    def _run_executor(self):
        try:
            self.executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.terminate()

    def start(self):
        """Start spinning the nodes in a separate thread."""
        self.executor_thread = threading.Thread(target=self._run_executor)
        self.executor_thread.start()

    def terminate(self):
        """Terminate all nodes and shutdown rclpy."""
        for node in self.nodes:
            node.destroy_node()
        rclpy.shutdown()
        if self.executor_thread:
            self.executor_thread.join()

class GO2Real(Node):
    def __init__(
        self,
        mode = 'highlevel', # 'highlevel' or 'lowlevel'
        vx_max=0.5,
        vy_max=0.4,
        ωz_max=0.5,
    ):
        assert mode in ['highlevel', 'lowlevel'], "mode should be either 'highlevel' or 'lowlevel'"
        self.simulated = False
        self.prestanding_q = np.array([ 0.0,  1.26186061, -2.5,
                                    0.0,  1.25883281, -2.5,
                                    0.0,  1.27193761, -2.6,  
                                    0.0,  1.27148342, -2.6])

        self.sitting_q = np.array([-0.02495611,  1.26249647, -2.82826662,
                                    0.04563564,  1.2505368 , -2.7933557 ,
                                   -0.30623949,  1.28283751, -2.82314873,  
                                    0.26400229,  1.29355574, -2.84276843])

        self.standing_q = np.array([ 0.0,  0.77832842, -1.56065452,
                                     0.0,  0.76754963, -1.56634164,
                                     0.0,  0.76681757, -1.53601146,  
                                     0.0,  0.75422204, -1.53229916])
        self.latest_command_stamp = time.time()
        self.mode = mode
        self.node_name = "go2py_highlevel_subscriber"
        self.highcmd_topic = "/go2/twist_cmd"
        self.lowcmd_topic = "/go2/lowcmd"
        self.joint_state_topic = "/go2/joint_states"
        self.lowstate_topic = "/lowstate"
        super().__init__(self.node_name)
        
        self.lowstate_subscriber = self.create_subscription(
            LowState, self.lowstate_topic, self.lowstate_callback, 1
        )
        self.lowcmd_publisher = self.create_publisher(Go2pyLowCmd, self.lowcmd_topic, 1)

        self.odometry_subscriber = self.create_subscription(
            Odometry, "/utlidar/robot_odom", self.odom_callback, 1
        )

        self.highcmd_publisher = self.create_publisher(TwistStamped, self.highcmd_topic, 1)
        self.highcmd = TwistStamped()
        # create pinocchio robot
        # self.pin_robot = PinRobot()

        # for velocity clipping
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.P_v_max = np.diag([1 / self.vx_max**2, 1 / self.vy_max**2])
        self.ωz_max = ωz_max
        self.ωz_min = -ωz_max
        self.running = True
        self.setCommands = {'lowlevel':self.setCommandsLow,
                            'highlevel':self.setCommandsHigh}[self.mode]
        self.state = None

    def lowstate_callback(self, msg):
        """
        Retrieve the state of the robot
        """
        self.state = msg

    def odom_callback(self, msg):
        """
        Retrieve the odometry of the robot
        """
        self.odom = msg

    def getOdometry(self):
        """Returns the odometry of the robot"""
        stamp = self.odom.header.stamp
        position = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        orientation = np.array([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w])
        stamp_nanosec = stamp.sec + stamp.nanosec * 1e-9
        return {'stamp_nanosec':stamp_nanosec, 'position':position, 'orientation':orientation}

    def getIMU(self):
        accel = self.state.imu_state.accelerometer
        gyro = self.state.imu_state.gyroscope
        quat = self.state.imu_state.quaternion
        rpy = self.state.imu_state.rpy
        temp = self.state.imu_state.temperature
        # return accel, gyro, quat, temp
        return {'accel':accel, 'gyro':gyro, 'quat':quat, "rpy":rpy, 'temp':temp}

    def getFootContacts(self):
        """Returns the raw foot contact forces"""
        footContacts = self.state.foot_force
        return np.array(footContacts)

    def getJointStates(self):
        """Returns the joint angles (q) and velocities (dq) of the robot"""
        if self.state is None:
            return None
        motor_state = np.array([[self.state.motor_state[i].q,
                                 self.state.motor_state[i].dq,
                                 self.state.motor_state[i].ddq,
                                 self.state.motor_state[i].tau_est,
                                 self.state.motor_state[i].temperature] for i in range(12)])
        return {'q':motor_state[:,0], 
                'dq':motor_state[:,1],
                'ddq':motor_state[:,2],
                'tau_est':motor_state[:,3],
                'temperature':motor_state[:,4]}

    def getRemoteState(self):
        """A method to get the state of the wireless remote control. 
        Returns a xRockerBtn object: 
        - head: [head1, head2]
        - keySwitch: xKeySwitch object
        - lx: float
        - rx: float
        - ry: float
        - L2: float
        - ly: float
        """
        wirelessRemote = self.state.wireless_remote[:24]

        binary_data = bytes(wirelessRemote)

        format_str = "<2BH5f"
        data = struct.unpack(format_str, binary_data)

        head = list(data[:2])
        lx = data[3]
        rx = data[4]
        ry = data[5]
        L2 = data[6]
        ly = data[7]

        _btn = bin(data[2])[2:].zfill(16)
        btn = [int(char) for char in _btn]
        btn.reverse()

        keySwitch = xKeySwitch(*btn)
        rockerBtn = xRockerBtn(head, keySwitch, lx, rx, ry, L2, ly)
        return rockerBtn

    def getCommandFromRemote(self):
        """Do not use directly for control!!!"""
        rockerBtn = self.getRemoteState()

        lx = rockerBtn.lx
        ly = rockerBtn.ly
        rx = rockerBtn.rx

        v_x = ly * self.vx_max
        v_y = lx * self.vy_max
        ω = rx * self.ωz_max
        
        return v_x, v_y, ω

    def getBatteryState(self):
        """Returns the battery percentage of the robot"""
        batteryState = self.state.bms
        return batteryState.SOC

    def setCommandsHigh(self, v_x, v_y, ω_z, bodyHeight=0.0, footRaiseHeight=0.0, mode=2):
        self.cmd_watchdog_timer = time.time()
        _v_x, _v_y, _ω_z = self.clip_velocity(v_x, v_y, ω_z)
        self.highcmd.header.stamp = self.get_clock().now().to_msg()
        self.highcmd.header.frame_id = "base_link"
        self.highcmd.twist.linear.x = _v_x
        self.highcmd.twist.linear.y = _v_y
        self.highcmd.twist.angular.z = _ω_z
        self.highcmd_publisher.publish(self.highcmd)

    def setCommandsLow(self, q_des, dq_des, kp, kd, tau_ff):
        # assert q_des.size == dq_des.size == kp.size == kd.size == tau_ff.size == 12, "q, dq, kp, kd, tau_ff should have size 12"
        lowcmd = Go2pyLowCmd()
        lowcmd.q = q_des
        lowcmd.dq = dq_des
        lowcmd.kp = kp
        lowcmd.kd = kd
        lowcmd.tau = tau_ff
        self.lowcmd_publisher.publish(lowcmd)
        self.latest_command_stamp = time.time()


    def close(self):
        self.running = False
        self.thread.join()
        self.destroy_node()

    def check_calf_collision(self, q):
        self.pin_robot.update(q)
        in_collision = self.pin_robot.check_calf_collision(q)
        return in_collision

    def clip_velocity(self, v_x, v_y, ω_z):
        _v = np.array([[v_x], [v_y]])
        _scale = np.sqrt(_v.T @ self.P_v_max @ _v)[0, 0]

        if _scale > 1.0:
            scale = 1.0 / _scale
        else:
            scale = 1.0

        return scale * v_x, scale * v_y, np.clip(ω_z, self.ωz_min, self.ωz_max)

    def overheat(self):
        return False

    def getGravityInBody(self):
        q = self.getIMU()['quat']
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        g_in_body = R.T@np.array([0.0, 0.0, -1.0]).reshape(3, 1)
        return g_in_body

class ROS2TFInterface(Node):

    def __init__(self, parent_name, child_name, node_name):
        super().__init__(f'{node_name}_tf2_listener')
        self.parent_name = parent_name
        self.child_name = child_name
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.T = None
        self.stamp = None
        self.running = True
        self.thread = threading.Thread(target=self.update_loop)
        self.thread.start()
        self.trans = None

    def update_loop(self):
        while self.running:
            try:
                self.trans = self.tfBuffer.lookup_transform(self.parent_name, self.child_name, rclpy.time.Time(), rclpy.time.Duration(seconds=0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                pass
            time.sleep(0.01)    

    def get_pose(self):
        if self.trans is None:
            return None
        else:
            translation = [self.trans.transform.translation.x, self.trans.transform.translation.y, self.trans.transform.translation.z]
            rotation = [self.trans.transform.rotation.x, self.trans.transform.rotation.y, self.trans.transform.rotation.z, self.trans.transform.rotation.w]
            self.T = np.eye(4)
            self.T[0:3, 0:3] = R.from_quat(rotation).as_matrix()
            self.T[:3, 3] = translation
            self.stamp = self.trans.header.stamp.nanosec * 1e-9 + self.trans.header.stamp.sec
            return self.T

    def close(self):
        self.running = False
        self.thread.join()  
        self.destroy_node()