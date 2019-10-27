
import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1

class Controller(object):
    def __init__(self, vehicle_mass, fuel_cap, brake_deadband, decel_lmt,
    	accel_lmt, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        self.yaw_ctrl = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0 # Minimum throttle value
        mx = 0.2 # Maximum throttle value

        self.throttle_ctrl = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # 1/(2*pi*tau) = cutoff frequency
        ts = 0.02 # sample time

        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_cap = fuel_cap
        self.brake_deadband = brake_deadband
        self.decel_lmt = decel_lmt
        self.accel_lmt = accel_lmt
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.last_vel = 0

    def control(self, current_vel, angular_vel, linear_vel, dbw_enabled):
        # Return throttle, brake, steer
        if not dbw_enabled:
        	self.throttle_ctrl.reset()
        	return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_ctrl.get_steering(linear_vel, angular_vel, current_vel)
        vel_err = linear_vel - current_vel
        self.last_vel = current_vel

        curr_time = rospy.get_time()
        sample_time = curr_time - self.last_time
        self.last_time = curr_time

        throttle = self.throttle_ctrl.step(vel_err, sample_time)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
        	throttle = 0
        	brake = 700 # to hold the car in place
        elif throttle < .1 and vel_err < 0:
        	throttle = 0
        	decel = max(vel_err, self.decel_lmt)
        	brake = abs(decel)*self.vehicle_mass*self.wheel_radius #Torqur N*m

        return throttle, brake,steering
