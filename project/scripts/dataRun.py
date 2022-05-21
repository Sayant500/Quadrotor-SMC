#!/usr/bin/env python3
# from math import pi, sqrt, atan2, cos, sin
from turtle import position
import numpy as np
from numpy import NaN
import rospy
import tf
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, Pose2D
import pickle
import os

class Quadrotor():
    def __init__(self):
        # publisher for rotor speeds
        self.motor_speed_pub = rospy.Publisher("/crazyflie2/command/motor_speed", Actuators, queue_size=10)
        # subscribe to Odometry topic
        self.odom_sub = rospy.Subscriber("/crazyflie2/ground_truth/odometry", Odometry, self.odom_callback)
        self.t0 = None
        self.t = None
        self.t_series = []
        self.x_series = []
        self.y_series = []
        self.z_series = []
        self.mutex_lock_on = False
        rospy.on_shutdown(self.save_data)

        # TODO: include initialization codes if needed

        self.omega = np.array([[0], [0], [0], [0]])
        self.m = 27e-3
        self.l = 46e-3
        self.Ix = 16.571710e-6
        self.Iy = 16.571710e-6
        self.Iz = 29.261652e-6
        self.Ip = 12.65625e-8
        self.kF = 1.28192e-8
        self.kM = 5.964552e-3
        self.omega_max = 2618
        self.omega_min = 0
        self.pos_d = np.zeros((3, 1))
        self.vel_d = np.zeros((3, 1))
        self.acc_d = np.zeros((3, 1))
        self.g = 9.81
        self.t_traj = 0

        self.ori_d = np.zeros((3, 1))
        self.dori_d = np.zeros((3, 1))
        self.ddori_d = np.zeros((3, 1))

    def gen_traj(self, t_d, coord_d):

        t0 = 0  # t_d[0]
        tf = t_d  # t_d[1]

        pos = np.hstack((coord_d[:, 0], coord_d[:, 1]))

        Mtx_t = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                          [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                          [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                          [1, tf, tf**2, tf**3, tf**4, tf**5],
                          [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                          [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])

        x = np.array([[coord_d[0, 0]], [0], [0], [coord_d[0, 1]], [0], [0]])
        y = np.array([[coord_d[1, 0]], [0], [0], [coord_d[1, 1]], [0], [0]])
        z = np.array([[coord_d[2, 0]], [0], [0], [coord_d[2, 1]], [0], [0]])

        ax = np.dot(np.linalg.inv(Mtx_t), x)
        ay = np.dot(np.linalg.inv(Mtx_t), y)
        az = np.dot(np.linalg.inv(Mtx_t), z)

        return ax, ay, az


    def traj_evaluate(self):

        t = np.array([0, 5, 20, 35, 50, 65])

        x = np.array([0, 0, 1, 1, 0, 0])
        y = np.array([0, 0, 0, 1, 1, 0])
        z = np.array([0, 1, 1, 1, 1, 1])

        coord = np.vstack((x, y, z))

        if (self.t >= t[0]) & (self.t <= t[1]):
            t_d = 5
            coord_d = coord[:, 0:2]
            self.t_traj = self.t - t[0]
            ax, ay, az = self.gen_traj(t_d, coord_d)
        elif (self.t > t[1]) & (self.t <= t[2]):
            t_d = 15
            coord_d = coord[:, 1:3]
            self.t_traj = self.t - t[1]
            ax, ay, az = self.gen_traj(t_d, coord_d)
        elif (self.t > t[2]) & (self.t <= t[3]):
            t_d = 15
            coord_d = coord[:, 2:4]
            self.t_traj = self.t - t[2]
            ax, ay, az = self.gen_traj(t_d, coord_d)
        elif (self.t > t[3]) & (self.t <= t[4]):
            t_d = 15
            coord_d = coord[:, 3:5]
            self.t_traj = self.t - t[3]
            ax, ay, az = self.gen_traj(t_d, coord_d)
        elif (self.t > t[4]) & (self.t <= t[5]):
            t_d = 15
            coord_d = coord[:, 4:6]
            self.t_traj = self.t - t[4]
            ax, ay, az = self.gen_traj(t_d, coord_d)
        else:
            self.t_traj = self.t - t[5]
            ax = np.zeros((6, 1))
            ay = np.zeros((6, 1))
            az = np.zeros((6, 1))

        t_iter = np.array([[1, self.t_traj, self.t_traj**2, self.t_traj**3, self.t_traj**4, self.t_traj**5]])
        dt_iter = np.array([[0, 1, 2*self.t_traj, 3*self.t_traj**2, 4*self.t_traj**3, 5*self.t_traj**4]])
        ddt_iter = np.array([[0, 0, 2, 6*self.t_traj, 12*self.t_traj**2, 20*self.t_traj**3]])

        x_d = np.dot(t_iter, ax)
        y_d = np.dot(t_iter, ay)
        z_d = np.dot(t_iter, az)
        dx_d = np.dot(dt_iter, ax)
        dy_d = np.dot(dt_iter, ay)
        dz_d = np.dot(dt_iter, az)
        ddx_d = np.dot(ddt_iter, ax)
        ddy_d = np.dot(ddt_iter, ay)
        ddz_d = np.dot(ddt_iter, az)

        self.pos_d[0, 0] = x_d
        self.pos_d[1, 0] = y_d
        self.pos_d[2, 0] = z_d
        self.vel_d[0, 0] = dx_d
        self.vel_d[1, 0] = dy_d
        self.vel_d[2, 0] = dz_d
        self.acc_d[0, 0] = ddx_d
        self.acc_d[1, 0] = ddy_d
        self.acc_d[2, 0] = ddz_d

        if self.t < 0:
            self.pos_d[0, 0] = x[0]
            self.pos_d[1, 0] = y[0]
            self.pos_d[2, 0] = z[0]
        elif self.t > t[-1]:
            self.pos_d[0, 0] = x[-1]
            self.pos_d[1, 0] = y[-1]
            self.pos_d[2, 0] = z[-1]

        # TODO: evaluating the corresponding trajectories designed in Part 1 to
        #  return the desired positions, velocities and accelerations

    def wrapToPi(self, x):
        xwrap = np.remainder(x, 2 * np.pi)
        mask = np.abs(xwrap) > np.pi
        xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
        mask1 = x < 0
        mask2 = np.remainder(x, np.pi) == 0
        mask3 = np.remainder(x, 2 * np.pi) != 0
        xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
        return xwrap

    def enforce_limit(self, data, max_value, min_value):
        for i in range(len(data)):
            if data[i] >= max_value:
                data[i] = max_value
            if data[i] <= min_value:
                data[i] = min_value
        return data

    def enforce_limit2(self, data, max_value, min_value):
        if (abs(data) > max_value).sum() > 0:
            data = data / np.max(abs(data)) * max_value

        for i in range(len(data)):
            if data[i] >= max_value:
                data[i] = max_value
            if data[i] <= min_value:
                data[i] = min_value
        return data

    def saturation(self, s, bl_width):
        return np.minimum(np.maximum(s/bl_width, -1), 1)

    def smc_control(self, xyz, xyz_dot, rpy, rpy_dot):

        # obtain the desired values by evaluating the corresponding trajectories

        self.traj_evaluate()

        # TODO: implement the Sliding Mode Control laws designed in Part 2 to calculate the control
        #  inputs "u"
        # REMARK: wrap the roll-pitch-yaw angle errors to [-pi to pi]

        d_rpy = np.zeros((3, 1))
        d_rpy_dot = np.zeros((3, 1))
        d_rpy_ddot = np.zeros((3, 1))

        # TODO: implement the Sliding Mode Control laws designed in Part 2 to calculate the control
        #  inputs "u"
        # REMARK: wrap the roll-pitch-yaw angle errors to [-pi to pi]

        #################################### Parameters #######################################################
        # k1 = 40 # 40 # 10
        # k2 = 20 # 80  #  80
        # k3 = 20 # 80 #  80
        # k4 = 20 # 80  # 80

        k1 = 50
        k2 = 92.5
        k3 = 92.5
        k4 = 20

        lambda1 = 10 #4 # 4
        lambda2 = 12 #9 # 9
        lambda3 = 12 #9 # 9
        lambda4 = 20 #9 # 9

        # lambda1 = 4
        # lambda2 = 9
        # lambda3 = 9
        # lambda4 = 9

        # Boundary layer width
        # bl_width_linear = 8e-1
        # bl_width = 2e0

        bl_width= 0.95 # 8e-1

        # K = np.array([[135, 0, 10, 0], [0, 135, 0, 10]])

        # K = np.array([[12, 0, 7, 0], [0, 12, 0, 7]])

        # K = np.array([[110, 0, 2, 0], [0, 110, 0, 2]])

        K = np.array([[33, 0, 9, 0], [0, 33, 0, 9]])

        # K = np.array([[72, 0, 17, 0], [0, 72, 0, 17]])

        e_xy = np. array([[xyz[0, 0] - self.pos_d[0, 0]], [xyz[1, 0] - self.pos_d[1, 0]], [xyz_dot[0, 0] - self.vel_d[0, 0]],
                          [xyz_dot[1, 0] - self.vel_d[1, 0]]])

        F = self.m * (-np.dot(K, e_xy) + np.array([[self.acc_d[0, 0]], [self.acc_d[1, 0]]]))

        # print("F = %s" % F)




        ######################################### Sliding mode control for z  ##########################################################

        e_z = xyz[2, 0] - self.pos_d[2, 0]
        e_zdot = xyz_dot[2, 0] - self.vel_d[2, 0]
        s_z = e_zdot + lambda1 * e_z
        denom = (np.cos(rpy[0, 0]) * np.cos(rpy[1, 0])) / self.m

        u_1R = -k1 * self.saturation(s_z, bl_width)
        # u_1R = -k1 * np.sign(s_z)

        ############### Control law ###############################
        u_1 = -(lambda1 * e_zdot - self.acc_d[2, 0] - self.g) / denom + u_1R / denom


        ######################################### Desired r (phi) and p (theta) from x and y ##########################################
        # print('d_rpy = %s' % d_rpy)
        # print('real_rpy = %s' % rpy)

        d_rpy[0, 0] = np.arcsin(- F[1, 0] / u_1)
        d_rpy[1, 0] = np.arcsin(F[0, 0] / u_1)


        ###### Calculate Omega

        Omega = self.omega[0] - self.omega[1] + self.omega[2] - self.omega[3]

        ############################################# Sliding mode control for r (phi) #########################################

        f_r = rpy_dot[1, 0] * rpy_dot[2, 0] * (self.Iy - self.Iz) / self.Ix - (self.Ip / self.Ix) * Omega * rpy_dot[1, 0]
        g_r = 1 / self.Ix

        e_r = self.wrapToPi(np.array([rpy[0, 0] - d_rpy[0, 0]]))
        e_rdot = (rpy_dot[0, 0] - d_rpy_dot[0, 0])
        s_r = e_rdot + lambda2 * e_r
        u_2R = -k2 * self.saturation(s_r, bl_width)
        # u_2R = -k2 * np.sign(s_r)


        ############### Control law ###############################
        u_2 = -(lambda2 * e_rdot - d_rpy_ddot[0, 0] + f_r) / g_r + u_2R / g_r
        u_2 = u_2[0]

        # # print('u_2 *g_r = %s' % (u_2*g_r))
        # print('u2 = %s' % u_2)
        # print('f_r = %s' % f_r)
        # print('u2_other = %s' % (-(lambda2 * e_rdot - d_rpy_ddot[0, 0] + f_r)))
        # # # print('s_r = %s' % s_r)
        # # # # print('x = %s' % xyz[0,0])
        # # print('rpy= %s' % rpy[0, 0])
        # # print('rpy_d = %s' % d_rpy[0, 0])
        # # # print('e_rdot = %s' % e_rdot)
        # print('e_r = %s' % e_r)

        #################### Sliding mode control for p  (theta) #########################################

        f_p = rpy_dot[0, 0] * rpy_dot[2, 0] * (self.Iz - self.Ix) / self.Iy + self.Ip / self.Iy * Omega * rpy_dot[0, 0]
        g_p = 1 / self.Iy

        e_p = self.wrapToPi(np.array([rpy[1, 0] - d_rpy[1, 0]]))
        e_pdot = rpy_dot[1, 0]-d_rpy_dot[1, 0]
        s_p = e_pdot + lambda3 * e_p
        u_3R = -k3 * self.saturation(s_p, bl_width)
        # u_3R = -k3 * np.sign(s_p)

        ############### Control law ###############################
        u_3 = -(lambda3 * e_pdot - d_rpy_ddot[1, 0] + f_p) / g_p + u_3R / g_p
        u_3 = u_3[0]

        # print('u_3 = %s' % u_3)

        # print('u_2 *g_r = %s' % (u_2*g_r))
        # print('u3 = %s' % u_3)
        # print('dtheta_d = %s' % d_rpy_dot[1,0])
        # print('f_p = %s' % -f_p)
        # print('u3_other = %s' % (-(lambda3 * e_pdot - d_rpy_ddot[1, 0] + f_p)))
        # # # print('s_r = %s' % s_r)
        # # # # print('x = %s' % xyz[0,0])
        # # print('rpy= %s' % rpy[0, 0])
        # # print('rpy_d = %s' % d_rpy[0, 0])
        # print('e_pdot = %s' % e_pdot)
        # print('e_p = %s' % e_p)

        ############################ Sliding mode control for y (psi) ##################################

        f_y = rpy_dot[0, 0] * rpy_dot[1, 0] * (self.Ix - self.Iy) / self.Iz
        g_y = 1 / self.Iz

        e_y = self.wrapToPi(np.array([rpy[2, 0] - d_rpy[2, 0]]))
        # print('e_y = %s' % e_y)
        e_ydot = rpy_dot[2, 0] - d_rpy_dot[2, 0]
        s_y = e_ydot + lambda3 * e_y
        u_4R = -k4 * self.saturation(s_y, bl_width)
        # u_4R = -k4 * np.sign(s_y)

        ############### Control law ###############################
        u_4 = -(lambda4 * e_ydot - d_rpy_ddot[2, 0] + f_y) / g_y + u_4R / g_y
        u_4 = u_4[0]

        # print('e_y = %s' % e_y)
        # print('u_4 = %s' % u_4)
        # print('u4_other = %s' % (-(lambda4 * e_ydot - d_rpy_ddot[2, 0] + f_y) / g_y))
        # print('s_y = %s' % s_y)
        # # print('x = %s' % xyz[0,0])
        # print('rpy= %s' % rpy)
        # print('rpy_d = %s' % d_rpy)
        # print('drpy = %s' % e_rdot)
        # print('e_r = %s' % e_r)

        # TODO: convert the desired control inputs "u" to desired rotor velocities "motor_vel" by
        #  using the "allocation matrix"

        alc_mat = np.array([[1 / (4 * self.kF), -np.sqrt(2) / (4 * self.kF * self.l),
                             -np.sqrt(2) / (4 * self.kF * self.l), -1 / (4 * self.kM * self.kF)],
                            [1 / (4 * self.kF), -np.sqrt(2) / (4 * self.kF * self.l),
                             np.sqrt(2) / (4 * self.kF * self.l), 1 / (4 * self.kM * self.kF)],
                            [1 / (4 * self.kF), np.sqrt(2) / (4 * self.kF * self.l),
                             np.sqrt(2) / (4 * self.kF * self.l), -1 / (4 * self.kM * self.kF)],
                            [1 / (4 * self.kF), np.sqrt(2) / (4 * self.kF * self.l),
                             -np.sqrt(2) / (4 * self.kF * self.l), 1 / (4 * self.kM * self.kF)]])

        omega_sq = np.dot(alc_mat, np.array([[u_1], [u_2], [u_3], [u_4]]).astype('float'))



        # TODO: maintain the rotor velocities within the valid range of [0 to 2618]

        # omega_sq = self.enforce_limit(omega_sq, self.omega_max**2, self.omega_min**2)

        # print('omega square = %s' % omega_sq)

        # omega_sq = self.enforce_limit2(omega_sq, self.omega_max ** 2, self.omega_min**2)
        self.omega = np.sqrt(omega_sq)

        # print('omega : %s' % self.omega)

        self.omega = self.enforce_limit(self.omega, self.omega_max, self.omega_min)

        # print('omega enforced : %s' % self.omega)

        motor_vel = self.omega
        # print(motor_vel)

        # publish the motor velocities to the associated ROS topic

        motor_speed = Actuators()
        motor_speed.angular_velocities = [motor_vel[0, 0], motor_vel[1, 0],
                                          motor_vel[2, 0], motor_vel[3, 0]]
        self.motor_speed_pub.publish(motor_speed)

    # odometry callback function (DO NOT MODIFY)

    def odom_callback(self, msg):
        if self.t0 == None:
            self.t0 = msg.header.stamp.to_sec()
        self.t = msg.header.stamp.to_sec() - self.t0

        # convert odometry data to xyz, xyz_dot, rpy, and rpy_dot
        w_b = np.asarray([[msg.twist.twist.angular.x], [msg.twist.twist.
                     angular.y], [msg.twist.twist.angular.z]])

        v_b = np.asarray([[msg.twist.twist.linear.x], [msg.twist.twist.linear.
                     y], [msg.twist.twist.linear.z]])

        xyz = np.asarray([[msg.pose.pose.position.x], [msg.pose.pose.position.
                     y], [msg.pose.pose.position.z]])

        q = msg.pose.pose.orientation

        T = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0:3, 3] = xyz[0:3, 0]
        R = T[0:3, 0:3]

        xyz_dot = np.dot(R, v_b)
        rpy = tf.transformations.euler_from_matrix(R, 'sxyz')
        rpy_dot = np.dot(np.asarray([
        [1, np.sin(rpy[0]) * np.tan(rpy[1]), np.cos(rpy[0]) * np.tan(rpy[1])],
        [0, np.cos(rpy[0]), -np.sin(rpy[0])],
        [0, np.sin(rpy[0]) / np.cos(rpy[1]), np.cos(rpy[0]) / np.cos(rpy[1])]]), w_b)

        rpy = np.expand_dims(rpy, axis=1)

        # store the actual trajectory to be visualized later
        if (self.mutex_lock_on is not True):
            self.t_series.append(self.t)
            self.x_series.append(xyz[0, 0])
            self.y_series.append(xyz[1, 0])
            self.z_series.append(xyz[2, 0])
            # call the controller with the current states
            self.smc_control(xyz, xyz_dot, rpy, rpy_dot)

    def save_data(self):
        with open("/home/jackzhy/rbe502_project/src/project/scripts/log.pkl", "wb") as fp:
            self.mutex_lock_on = True
            pickle.dump([self.t_series, self.x_series, self.y_series, self.z_series], fp)



if __name__ == '__main__':
    rospy.init_node("quadrotor_control")
    rospy.loginfo("Press Ctrl + C to terminate")
    dataRun = Quadrotor()
    # rate =rospy.Rate(0.01)
    try:
        # rate.sleep()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")