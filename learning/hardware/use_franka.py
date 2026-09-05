from franka_control import FrankaController
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--gripper", default = True)
    parser.add_argument("--homing", action="store_true")
    args = parser.parse_args()

    franka = FrankaController(
        robot_ip=args.ip,
        use_gripper=args.gripper,
        do_gripper_homing=args.homing,
        realtime=False,
    )


    # # the main functions (same API as UR5eRobot / FrankaRobot):

    # franka.get_tcp_pose()               # (6,) [x,y,z,rx,ry,rz] m/rad (rotation vector)
    # franka.get_joint_angles()           # (7,) joint angles rad
    # franka.get_gripper()                # width of gripper
    # franka.gripper_move(pos, speed)
    # franka.gripper_grasp(pos, speed, force)
    # franka.move_tcp_pose(target_pose, velocity, acceleration, asynchronous)   # (6,) target
    # franka.set_ee_velocity(linear_velocity, angular_velocity, max_vel, max_ang_vel)  # real-time, e.g. spacemouse
    # franka.move_joints(target_joints, velocity, acceleration, asynchronous)   # (7,) target
    # franka.stop()        # stop motion/servo, keep connection alive
    # franka.recover()     # automatic error recovery
    # franka.disconnect()  # stop and release



    # Historical 16-value O_T_EE targets (pre-refactor). The public API now
    # takes 6D [x,y,z,rx,ry,rz] poses via move_tcp_pose();
    # convert with FrankaController._O_T_EE_to_pose6(target) before reusing.

    #left pose
#     target = [ 0.78287947, -0.44356015, -0.43629587,  0.  ,        0.37657288, -0.22040279,
#   0.89978635,  0.,         -0.49527022, -0.86872149, -0.00551611,  0.
# ,  0.2844739,  -0.35223967,  0.53286338,  1.        ]
      #bottom pose
#     target = [ 0.99772137, -0.01136652, -0.06650478 , 0.0  ,       -0.01270037, -0.99972588,
#      -0.01966809,  0.0,         -0.066263,    0.02046791, -0.99759227,  0.,
#   0.62270474,  0.10761493,  0.32549337,  1.0 ]
#     franka.move_tcp_pose(FrankaController._O_T_EE_to_pose6(target))



if __name__ == "__main__":
    main()
