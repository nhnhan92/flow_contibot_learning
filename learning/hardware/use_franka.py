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
        ip=args.ip,
        use_gripper=args.gripper,
        do_gripper_homing=args.homing,
        realtime=False,
    )


    # # the main functions:
    
    # franka.get_ee # (16 component) / x,y z is 12,13,14 order in this object pose
    # franka.get_joint # 7 component joint
    # franka.get_gripper # width of gripper 
    # franka.gripper_move(pos,speed) 
    # franka.gripper_grasp(pos,speed, force)
    # franka.move_ee(16 component) / x,y z is 12,13,14 order in this object pose
    # franka.move_joint(7 component joint )   
    # franka.stop()  # force stop all



    
    #left pose
#     target = [ 0.78287947, -0.44356015, -0.43629587,  0.  ,        0.37657288, -0.22040279,
#   0.89978635,  0.,         -0.49527022, -0.86872149, -0.00551611,  0.
# ,  0.2844739,  -0.35223967,  0.53286338,  1.        ]
      #bottom pose
#     target = [ 0.99772137, -0.01136652, -0.06650478 , 0.0  ,       -0.01270037, -0.99972588,
#      -0.01966809,  0.0,         -0.066263,    0.02046791, -0.99759227,  0.,
#   0.62270474,  0.10761493,  0.32549337,  1.0 ]
#     franka.move_ee(target)



if __name__ == "__main__":
    main()