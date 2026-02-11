import serial
import csv
import time
from datetime import datetime
import threading
import matplotlib.pyplot as plt
import os
SERIAL_PORT = "COM9"
BAUD_RATE   = 115200
OUTPUT_CSV  = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

MAX_POINTS = 1000

def reader_logger(ser, writer, t_vals, flow_vals, press_vals, stop_flag,save_fig, name):

    # Read header
    header_line = ser.readline().decode("utf-8", errors="ignore").strip()
    print("Header:", header_line)
    if header_line.startswith("t_ms"):
        header = header_line.split(",")
    else:
        header = ["current_time","t_ms","raw_flow","flow_Lmin","raw_press","press_MPa",
                  "pwm1_cur","pwm2_cur","pwm3_cur"]
    writer.writerow(header)

    t0 = None
    prev_t_ms = None

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    line_flow,  = ax1.plot([], [], label="Flow [L/min]")
    line_press, = ax2.plot([], [], label="Pressure [MPa]")
    ax1.set_ylabel("Flow [L/min]")
    ax2.set_ylabel("Pressure [MPa]")
    ax2.set_xlabel("Time [s]")
    ax1.legend(); ax2.legend()

    while not stop_flag["stop"]:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S.%f")
        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 5:
            continue
        written_data = [current_time] + parts
        writer.writerow(written_data)
        try:
            t_ms      = float(parts[0])
            flow_lpm  = float(parts[2])
            press_mpa = float(parts[4])
        except ValueError:
            continue

        if prev_t_ms is not None and t_ms < prev_t_ms:
            print("### Detected reset on Arduino, re-zeroing time.")
            t0 = None
            # Optionally also clear your buffers:
            t_vals.clear()
            flow_vals.clear()
            press_vals.clear()

        prev_t_ms = t_ms
        if t0 is None:
            t0 = t_ms
        t_s = (t_ms - t0) / 1000.0

        t_vals.append(t_s)
        flow_vals.append(flow_lpm)
        press_vals.append(press_mpa)
        if len(t_vals) > MAX_POINTS:
            t_vals[:]      = t_vals[-MAX_POINTS:]
            flow_vals[:]   = flow_vals[-MAX_POINTS:]
            press_vals[:]  = press_vals[-MAX_POINTS:]

        if len(t_vals) % 5 == 0:
            line_flow.set_data(t_vals, flow_vals)
            line_press.set_data(t_vals, press_vals)
            ax1.relim(); ax1.autoscale_view()
            ax2.relim(); ax2.autoscale_view()
            plt.pause(0.001)

    if save_fig:
        plt.savefig(f"{name}.png",format= "png",dpi=300, bbox_inches="tight")
        print("Figure saved to", f"{name}.png")
    plt.ioff()
    plt.show()
    

def main():
    import random
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, help='Choose object type',required=False)
    parser.add_argument('--module_no', '-n', type=int, help='Choose object type',default=1,required=False)

    args = parser.parse_args()
    mode = args.mode
    module_no = args.module_no

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)
    print("Opened", SERIAL_PORT)

    folder_path = f"data/{mode}"
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"log_module{module_no}_{mode}"
    OUTPUT_CSV = f"data/{mode}/{file_name}.csv"
    f = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f)

    t_vals, flow_vals, press_vals = [], [], []
    stop_flag = {"stop": False}
    save_fig = True
    # start reader logger thread
    t = threading.Thread(target=reader_logger,
                         args=(ser, writer, t_vals, flow_vals, press_vals, stop_flag,save_fig,file_name),
                         daemon=True)
    t.start()
    # print("Type 'A B C' to change PWM targets, 'quit' to stop.")
    try:
        while True:
            text = input(">> ").strip()
            if text.lower() in ("q", "quit", "exit"):
                cmd = "0 0 0\n"    
                ser.write(cmd.encode("ascii"))
                print("[PYTHON] Sent:", cmd.strip())
                break
            if not text:
                continue
            cmd = text + "\n"
            ser.write(cmd.encode("ascii"))
            print("[PYTHON] Sent:", text)
            if text.lower() in ("p"):  ## randomly drive to a pos
                # for _ in range(5):
                valve1_pwm = random.randint(5,20)
                valve2_pwm = random.randint(5,20)
                valve3_pwm = random.randint(5,20)
                cmd = f"{valve1_pwm} {valve2_pwm} {valve3_pwm}\n" 
                ser.write(cmd.encode("ascii"))
                print("[PYTHON] Sent:", cmd.strip())

            if text.lower() in ("single", "s") :  ## drive one module with different pwm 
                ## to measure the flowrate and pressure
                
                for ite in range(2):
                    TIME_STAMP = f"data/{mode}/{file_name}_timestamp_{ite}.csv"
                    f_timestamp = open(TIME_STAMP, "w", newline="")
                    writer_timestamp = csv.writer(f_timestamp)
                    header_timestamp = ["current_time","pwm_incr"]
                    writer_timestamp.writerow(header_timestamp)
                    for i in range(1, 26):
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        cmd = f"{i} 0 0\n"
                        ser.write(cmd.encode("ascii"))
                        print("[PYTHON] Sent:", cmd.strip())
                        time.sleep(1.5)  
                        data = [current_time, i]
                        writer_timestamp.writerow(data)
                    cmd = "0 0 0\n"    
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                    time.sleep(4) 
                break
            if text.lower() in ("double", "d") :  ## drive one module with different pwm 
                ## to measure the flowrate and pressure
                file_name = f"log_module{module_no}vs3_{mode}"
                TIME_STAMP = f"data/{mode}/{file_name}_timestamp_{ite}.csv"
                f_timestamp = open(TIME_STAMP, "w", newline="")
                writer_timestamp = csv.writer(f_timestamp)
                header_timestamp = ["current_time","pwm_incr_module1","pwm_incr_module2"]
                writer_timestamp.writerow(header_timestamp)
                for module_2 in range(1, 26):
                    for module_1 in range(1,26):
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        cmd = f"{module_1} 0 {module_2}\n"
                        ser.write(cmd.encode("ascii"))
                        print("[PYTHON] Sent:", cmd.strip())
                        time.sleep(1.5)  
                        data = [current_time, module_1,module_2]
                        writer_timestamp.writerow(data)
                    cmd = "0 0 0\n"    
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                    time.sleep(4)   
                break
            if text.lower() in ("triple", "t") :  
                ## to measure the flowrate and pressure
                file_name = f"log_module{mode}"
                TIME_STAMP = f"data/{mode}/{file_name}_timestamp_{ite}.csv"
                f_timestamp = open(TIME_STAMP, "w", newline="")
                writer_timestamp = csv.writer(f_timestamp)
                header_timestamp = ["current_time","pwm_incr_module1","pwm_incr_module2","pwm_incr_module3"]
                writer_timestamp.writerow(header_timestamp)
                
                pairs = []   
                for module_1 in range(1, 26):
                    for module_2 in range(1,26):
                        for module_3 in range(module_2,26):
                            pairs.append(module_1,module_2,module_3)
                count = 20
                ite = int(len(pairs)/count)
                for i in range(0,ite):
                    for j in range(count):
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        cmd = f"{pairs[count*i+j][0]} {pairs[count*i+j][1]} {pairs[count*i+j][2]}\n"
                        ser.write(cmd.encode("ascii"))
                        print("[PYTHON] Sent:", cmd.strip())
                        time.sleep(1.5)  
                        data = [current_time, pairs[count*i+j][0],pairs[count*i+j][1],pairs[count*i+j][2]]
                        writer_timestamp.writerow(data)
                    print("SLEEPINGGGGGGGGGGGG")
                    cmd = "0 0 0\n"    
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                    time.sleep(4)
                    
                for i in range(len(pairs[:count*ite]),len(pairs[0:25])):
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    cmd = f"{pairs[i][0]} {pairs[i][1]} {pairs[i][2]}\n"
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                    time.sleep(1.5)  
                    data = [current_time, pairs[i][0],pairs[i][1],pairs[i][2]]
                    writer_timestamp.writerow(data)
                            
                    cmd = "0 0 0\n"    
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                    time.sleep(4)   
                break
                    
                    
    except KeyboardInterrupt:
        pass

    stop_flag["stop"] = True
    t.join(timeout=2.0)

    f.flush()
    f.close()
    ser.close()
    print("Closed serial, data saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
