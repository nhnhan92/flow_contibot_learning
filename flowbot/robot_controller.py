from kinematic_modeling import Flow_driven_bellow
from workspace import workspace_using_fwdmodel
import numpy as np
import random
import argparse
import serial
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import threading

SERIAL_PORT = "COM9"
BAUD_RATE   = 115200

def drain_serial(ser, stop_flag):
    while not stop_flag["stop"]:
        try:
            line = ser.readline()  # read and discard (or print)
            # print(line.decode("utf-8", errors="ignore").strip())
        except Exception:
            pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, help='Choose object type',required=False)
    parser.add_argument('--module_no', '-n', type=int, help='Choose object type',default=1,required=False)

    args = parser.parse_args()
    mode = args.mode
    module_no = args.module_no

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1,write_timeout=1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)
    print("Opened", SERIAL_PORT)
    
    stop_flag = {"stop": False}
    t = threading.Thread(target=drain_serial, args=(ser, stop_flag), daemon=True)
    t.start()

    robot = Flow_driven_bellow(
            D_in = 5,
            D_out = 16.5,
            l0=82,
            d=28.17,
            lb=0.0,
            lu=13.5,
            k_model= lambda deltal: 0.18417922367667078 + 0.1511268093994831 * (1.0 - np.exp(-0.18801952663756039 * deltal)),
            a_delta = 0,
            b_delta= 0,
            a_pwm2press= 0.004227,
            b_pwm2press= 0.012059,
        )
    workspace = workspace_using_fwdmodel(robot=robot,pwm_min=1, pwm_max=20)
    hull = workspace.build_workspace_hull_checker(workspace.P)
    tri = hull["tri"]
    bbox = hull["bbox"]
    try:
        while True:
            text = input(">> ").strip()
            if not text:
                continue
            if text.lower() in ("i"):  ## randomly drive to a pos with inverse model
                
                p = workspace.sample_random_point_in_workspace(tri, bbox)
                # print(f"Moving to [{p[0]} {p[1]} {p[2]}]")
                ik = robot.inverse_pressures_from_position(p)
                # print(f"Pressure = [{ik['pb'][0]} {ik['pb'][1]} {ik['pb'][2]}]")
                # print(f"PWM signal = [{ik['pwm'][0]} {ik['pwm'][1]} {ik['pwm'][2]}]")
                cmd = f"{ik['pwm'][0]} {ik['pwm'][1]} {ik['pwm'][2]}\n" 
                ser.write(cmd.encode("ascii"))
                print("[PYTHON] Sent:", cmd.strip())

            if text.lower() in ("f"):  ## randomly drive to a pos with forward model
                valve1_pwm = random.randint(5,20)
                valve2_pwm = random.randint(5,20)
                valve3_pwm = random.randint(5,20)
                pwm_signals = np.array([valve1_pwm, valve2_pwm, valve3_pwm], dtype=int)
                print(f"PWM signal = [{valve1_pwm} {valve2_pwm} {valve3_pwm}]")
                pb = robot.pwm_to_pressure(pwm=pwm_signals)
                print(f"pressure = {pb}")
                fk = robot.forward_kinematics_from_pressures(pb)
                print("Forward pc:", fk["pc"], "lengths:", fk["l"])

                cmd = f"{valve1_pwm} {valve2_pwm} {valve3_pwm}\n" 
                ser.write(cmd.encode("ascii"))
                print("[PYTHON] Sent:", cmd.strip())

            if text.lower() in ("q", "quit", "exit"):
                cmd = "0 0 0\n"    
                ser.write(cmd.encode("ascii"))
                print("[PYTHON] Sent:", cmd.strip())
                break
    except KeyboardInterrupt:
        pass
    stop_flag["stop"] = True
    ser.close()
    print("Closed serial")

if __name__ == "__main__":
    main()