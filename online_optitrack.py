from __future__ import annotations

import csv
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from natnet import NatNetClient, DataFrame  # pip install natnet


@dataclass(frozen=True)
class RigidBodySample:
    t_arrival: float          # time.perf_counter() when received
    frame_number: int
    natnet_timestamp: Optional[float]  # may be None depending on server/version
    pos_xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]

    

class MotiveNatNetReader:
    """
    Receives NatNet frames in background threads and keeps the latest rigid body pose.
    """
    def __init__(
        self,
        server_ip: str,
        local_ip: str,
        use_multicast: bool,
        rigid_body_id: int,
    ):
        self.server_ip = server_ip
        self.local_ip = local_ip
        self.use_multicast = use_multicast
        self.rigid_body_id = rigid_body_id

        self._lock = threading.Lock()
        self._latest: Optional[RigidBodySample] = None
        self.R_MW = np.array([[0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0]])

        self._client = NatNetClient(
            server_ip_address=self.server_ip,
            local_ip_address=self.local_ip,
            use_multicast=self.use_multicast,
        )
        self._client.on_data_frame_received_event.handlers.append(self._on_frame)
    def Rz(self,alpha):
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([[ca, -sa, 0.0],
                        [sa,  ca, 0.0],
                        [0.0, 0.0, 1.0]])

    

    def opti_to_manip(self,pos_W_m, origin_W_m, alpha_rad, scale=1000.0):
        pW = np.array(pos_W_m, dtype=float)
        p0 = np.array(origin_W_m, dtype=float)
        p_rel = pW - p0
        pM = self.Rz(alpha_rad) @ (self.R_MW @ p_rel)
        return pM * scale  # into mm if scale=1000

    def _on_frame(self, data_frame):
        t_arrival = time.perf_counter()

        if not data_frame.rigid_bodies:
            return

        # If you want a specific rigid body by streaming ID:
        rb = None
        for r in data_frame.rigid_bodies:
            if r.id_num == self.rigid_body_id:
                rb = r
                break

        # If not found, do nothing
        if rb is None:
            return

        # Optional: ignore invalid tracking frames
        if hasattr(rb, "tracking_valid") and (not rb.tracking_valid):
            return

        frame_number = getattr(data_frame, "frame_number", -1)
        natnet_ts = getattr(data_frame, "timestamp", None)

        sample = RigidBodySample(
            t_arrival=t_arrival,
            frame_number=int(frame_number) if frame_number is not None else -1,
            natnet_timestamp=float(natnet_ts) if natnet_ts is not None else None,
            pos_xyz=(float(rb.pos[0]), float(rb.pos[1]), float(rb.pos[2])),
            quat_xyzw=(float(rb.rot[0]), float(rb.rot[1]), float(rb.rot[2]), float(rb.rot[3])),
        )

        with self._lock:
            self._latest = sample

    def start(self) -> None:
        # Connect and run async receiving threads
        self._client.connect()
        # Request model definitions (optional but useful for debugging IDs/names)
        self._client.request_modeldef()
        self._client.run_async()

    def stop(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            pass

    def get_latest(self) -> Optional[RigidBodySample]:
        with self._lock:
            return self._latest
        
    def parse_vec3(self,s: str) -> np.ndarray:
        """Parse 'x,y,z' into a (3,) numpy array."""
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError("Expected format 'x,y,z' (three numbers).")
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


    def opti_to_workspace_xyz(self,opti_pos_m: np.ndarray, origin_m: np.ndarray) -> np.ndarray:
        """Convert OptiTrack meters to workspace units and shift by predefined origin."""
        opti_pos_m = np.asarray(opti_pos_m, dtype=float).reshape(3,)
        origin_m = np.asarray(origin_m, dtype=float).reshape(3,)
        return (opti_pos_m - origin_m) * float(RigidBodySample)


class SyncCSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._f = open(csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "t_robot",
            "pc_x", "pc_y", "pc_z",
            "pwm_1", "pwm_2", "pwm_3",
            "motive_t_arrival",
            "motive_frame_number",
            "motive_timestamp",
            "rb_x", "rb_y", "rb_z",
            "rb_qx", "rb_qy", "rb_qz", "rb_qw",
        ])
        self._f.flush()

    def log(self, t_robot: float, pc_xyz, pwm_123, motive: Optional[RigidBodySample]) -> None:
        pc_x, pc_y, pc_z = [float(v) for v in pc_xyz]
        p1, p2, p3 = [int(v) for v in pwm_123]

        if motive is None:
            self._w.writerow([t_robot, pc_x, pc_y, pc_z, p1, p2, p3] + [None] * 9)
        else:
            self._w.writerow([
                t_robot,
                pc_x, pc_y, pc_z,
                p1, p2, p3,
                motive.t_arrival,
                motive.frame_number,
                motive.natnet_timestamp,
                *motive.pos_xyz,
                *motive.quat_xyzw,
            ])

        # flush occasionally (or every line if you prefer safety over speed)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass



def main():
    motive = MotiveNatNetReader(
    server_ip="192.168.11.15",
    local_ip="192.168.11.15",
    use_multicast=False,
    rigid_body_id=1,
)
    motive.start()

    logger = SyncCSVLogger("teleop_motive_sync.csv")
    try:
        while True:
            t_robot = time.perf_counter()
            motive_sample = motive.get_latest()
            print(motive_sample)
            # logger.log(t_robot=t_robot, pc_xyz=pc, pwm_123=pwm, motive=motive_sample)
    finally:
        motive.stop()
        # logger.close()

if __name__ == "__main__":
    main()
