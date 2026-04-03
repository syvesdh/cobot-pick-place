#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
import math

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

class MoveItXYZRPYClient(Node):
    def __init__(self):
        super().__init__('moveit_xyzrpy_client')
        # Binds to the standard MoveIt Action Server action
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        
    def send_ptp_goal(self, x, y, z, roll, pitch, yaw):
        self.get_logger().info('Waiting for MoveIt action server (/move_action)...')
        self._action_client.wait_for_server()
        
        goal_msg = MoveGroup.Goal()
        
        req = MotionPlanRequest()
        req.group_name = 'tmr_arm' # Standard TM MoveGroup name
        req.num_planning_attempts = 3
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        
        # Build Cartesion Constraints
        constraint = Constraints()
        constraint.name = "goal_xyzrpy"
        
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base" # Root frame
        pos_constraint.link_name = "flange" # End effector frame
        
        s = SolidPrimitive()
        s.type = SolidPrimitive.SPHERE
        s.dimensions = [0.01] # 1cm tolerance sphere
        
        bv = BoundingVolume()
        bv.primitives.append(s)
        
        ps = PoseStamped()
        ps.header.frame_id = "base"
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        bv.primitive_poses.append(ps.pose)
        
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0
        
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "base"
        ori_constraint.link_name = "flange"
        q = euler_to_quaternion(roll, pitch, yaw)
        ori_constraint.orientation.x = q[0]
        ori_constraint.orientation.y = q[1]
        ori_constraint.orientation.z = q[2]
        ori_constraint.orientation.w = q[3]
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0
        
        constraint.position_constraints.append(pos_constraint)
        constraint.orientation_constraints.append(ori_constraint)
        
        req.goal_constraints.append(constraint)
        goal_msg.request = req
        
        self.get_logger().info('Sending Coordinate PTP action to MoveIt...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by MoveIt! Is the point reachable?')
            return
        self.get_logger().info('Goal accepted, executing trajectory...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'MoveIt execution completed with status code: {result.error_code.val}')
        # Code 1 is SUCCESS!
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = MoveItXYZRPYClient()
    
    # Configure your desired XYZ (in meters) and RPY (in radians)
    action_client.send_ptp_goal(
        x=0.4, 
        y=0.1, 
        z=0.4, 
        roll=3.14159, 
        pitch=0.0, 
        yaw=0.0
    )
    
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
