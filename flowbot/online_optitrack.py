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


def _kabsch_rotation(P_ref: np.ndarray, P_meas: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Best-fit rotation R (3x3) such that P_meas[i] ~= R @ P_ref[i] + t, solved by
    the Kabsch algorithm (SVD of the cross-covariance of the centered point sets).
    Translation is implicitly removed by centering -- only rotation is returned.

    Args:
        P_ref, P_meas : (N, 3) arrays, N >= 3, matched point-to-point (same order).

    Returns:
        R              : (3, 3) rotation mapping P_ref's frame onto P_meas's frame.
        rms_resid_m    : RMS residual distance (same units as the inputs) after
                         applying R + the fitted translation -- a fit-quality metric.
        per_point_resid_m : (N,) residual distance for each individual point, same
                         order as P_ref/P_meas -- an outlier here (much larger than
                         the rest) points at which specific point's known/measured
                         position is wrong, rather than a generally noisy fit.
    """
    P_ref = np.asarray(P_ref, dtype=float)
    P_meas = np.asarray(P_meas, dtype=float)
    if P_ref.shape != P_meas.shape or P_ref.shape[0] < 3:
        raise ValueError(f"Need >=3 matched points of equal shape, got {P_ref.shape} vs {P_meas.shape}")

    c_ref = P_ref.mean(axis=0)
    c_meas = P_meas.mean(axis=0)
    A = P_ref - c_ref
    B = P_meas - c_meas

    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    t = c_meas - R @ c_ref
    resid = (R @ P_ref.T).T + t - P_meas
    per_point_resid_m = np.linalg.norm(resid, axis=1)
    rms_resid_m = float(np.sqrt(np.mean(per_point_resid_m ** 2)))
    return R, rms_resid_m, per_point_resid_m


def build_ring_marker_positions(
    marker_angles_deg: Dict[int, float],
    radius_m: float,
    z_offset_m: float = 0.0,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Build a local_marker_positions_m dict for markers arranged on a ring in the
    X_local-Y_local plane (perpendicular to Z_local), at a fixed Z_local offset
    -- matching the flowbot's distal-tip marker layout.

    Args:
        marker_angles_deg : {marker_id: angle_deg}, angle measured in the
                             X_local-Y_local plane (0 deg = +X_local axis,
                             increasing toward +Y_local -- adjust to match
                             however the angles were actually measured).
        radius_m           : ring radius (m) -- distance from the Z_local axis
                              to each marker.
        z_offset_m          : Z_local coordinate of the marker ring (m).

    Returns:
        {marker_id: (x, y, z)} in meters, local/manipulator frame.
    """
    out = {}
    for mid, ang_deg in marker_angles_deg.items():
        a = np.deg2rad(ang_deg)
        out[mid] = (radius_m * np.cos(a), radius_m * np.sin(a), z_offset_m)
    return out


# Flowbot distal-tip marker layout, keyed by marker index 1-6 (matching the
# physical layout diagram) -- this must match whatever key
# get_latest_markers() reports live for each marker (see its docstring).
#
# Built DIRECTLY from a real diagnose_marker_layout() run on the actual rig,
# rather than from the hand-drawn picture's angles -- the picture-derived
# angles (pairs 48.4deg apart, bisectors 120deg apart, radius=half of
# "122.23mm" which turned out to be a diameter not a radius) produced an
# error pattern against live data that was NOT a clean single-marker-swap or
# global-handedness-flip signature (4 markers at -45..-49deg error, marker 5
# at -94deg) -- i.e. still wrong in some way not worth further guessing.
# marker_angles_deg here are the LIVE MEASURED angles from the --diagnose
# table, shifted so 0deg = the {1,5} pair's bisector -- i.e. true local +X,
# per the original design (one axis per actuator, bisecting its marker
# pair) -- NOT marker #1's own position. A first attempt used marker #1
# itself as the 0deg reference; that gave a near-perfect shape fit
# (fit_rms_m dropped from ~60-90mm to 0.4mm) but alpha came out ~159deg
# instead of the expected ~+/-90deg, because marker #1 sits ~22.76deg off
# the true +X axis -- confirmed by checking that all three pair bisectors
# ({1,5}, {2,6}, {4,3}) are ~120deg apart in the raw data (within 0.6deg),
# so re-centering on the {1,5} bisector is the physically correct 0deg.
# radius_m is the live measured average radius (~61.2mm, confirming
# "122.23mm" was indeed a diameter: 122.23/2 = 61.115, matches within noise).
# z_offset_m=96.3mm is kept as given directly (not something this ring-plane
# diagnostic measures).
FLOWBOT_MARKER_POSITIONS_M: Dict[int, Tuple[float, float, float]] = build_ring_marker_positions(
    marker_angles_deg={
        1: 22.763,
        2: 97.513,
        3: -97.607,
        4: -143.537,
        5: -22.647,
        6: 143.513,
    },
    radius_m=0.061225,
    z_offset_m=0.0963,
)


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
        alpha: float = -3.1120913183568417,
        local_marker_positions_m: Optional[Dict[int, Tuple[float, float, float]]] = None,
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
        self.alpha = alpha

        # marker_id -> known position (m) in the manipulator/local frame (the
        # same frame opti_to_manip outputs into) -- set once from CAD/measurement
        # of where the 6 markers are rigidly mounted on the flowbot. Needed only
        # for calibrate_alpha_from_markers(); everything else works without it.
        self.local_marker_positions_m = local_marker_positions_m
        self._latest_markers: Dict[int, np.ndarray] = {}   # marker_id -> world pos (m)
        self._latest_markers_t: float = 0.0
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

    

    def opti_to_manip(self,pos_W_m, origin_W_m, scale=1000.0):
        pW = np.array(pos_W_m, dtype=float)
        p0 = np.array(origin_W_m, dtype=float)
        p_rel = pW - p0
        pM = self.Rz(self.alpha) @ (self.R_MW @ p_rel)
        return pM * scale  # into mm if scale=1000

    def _on_frame(self, data_frame):
        t_arrival = time.perf_counter()

        # Labeled markers belonging to our rigid body, keyed by marker_id (the
        # stable 1-6 per-asset marker index: id_num = model_id<<16 | marker_id,
        # confirmed against this project's actual hardware -- model_id equals
        # rigid_body_id, marker_id is 1-6, tied to the asset definition, not
        # reshuffled per-frame). Processed independently of the rigid-body pose
        # below (RigidBody.markers is None on modern NatNet protocols -- marker
        # data only comes via DataFrame.labeled_markers, not per-rigid-body).
        labeled = getattr(data_frame, "labeled_markers", None)
        if labeled:
            markers = {m.marker_id: np.array(m.pos, dtype=float)
                      for m in labeled if m.model_id == self.rigid_body_id}
            with self._lock:
                self._latest_markers = markers
                self._latest_markers_t = t_arrival

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
        print(f"Started NatNet client. Server: {self.server_ip}, Local: {self.local_ip}, Multicast: {self.use_multicast}, RigidBody ID: {self.rigid_body_id}")

    def stop(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            pass

    def get_latest(self) -> Optional[RigidBodySample]:
        with self._lock:
            return self._latest

    def get_latest_markers(self) -> Tuple[Dict[int, np.ndarray], float]:
        """
        Every currently-visible marker belonging to this rigid body: {marker_id
        (1-6): world pos (m)}, plus the perf_counter() timestamp they arrived
        at. Empty dict if none have been seen yet (e.g. labeled_markers
        unsupported by this NatNet protocol version, or no marker currently
        has model_id == rigid_body_id).
        """
        with self._lock:
            return dict(self._latest_markers), self._latest_markers_t

    def calibrate_alpha_from_markers(
        self,
        local_marker_positions_m: Optional[Dict[int, Tuple[float, float, float]]] = None,
        n_samples: int = 30,
        sample_interval_s: float = 0.05,
        timeout_s: float = 5.0,
        apply: bool = True,
    ) -> Dict[str, Any]:
        """
        Calibrate `alpha` from the flowbot's 6 markers, with no robot motion.

        Requires that the markers' positions in the manipulator/local frame
        (the frame opti_to_manip outputs into) are known -- e.g. from CAD or
        hand measurement -- and supplied either here or at construction time
        (local_marker_positions_m, see FLOWBOT_MARKER_POSITIONS_M). Format:
        {marker_id: (x, y, z) in meters} -- marker_id must match whatever key
        get_latest_markers() reports live for that marker.

        Algorithm: average n_samples live marker readings (decorrelated by
        sample_interval_s) per marker_id, then solve the best-fit rotation
        from local marker positions to their averaged world positions via
        Kabsch/Horn point-set registration (_kabsch_rotation) -- this is the
        FULL local->world rotation, independent of the existing alpha/R_MW.
        alpha is then extracted as the Z-axis angle such that
        Rz(alpha) @ R_MW best matches the world->local rotation implied by
        the fit, and the residual (both the Kabsch fit RMS and how far the
        implied rotation is from a pure Z-rotation) is reported as a health
        check -- large values suggest a bent/mislabeled marker or that R_MW
        itself no longer matches the physical setup.

        Returns a dict with keys: alpha_rad, alpha_deg, n_markers_used,
        marker_ids_used, fit_rms_m, z_rotation_deviation (0 = perfectly a
        pure Z-rotation; larger means something beyond alpha has changed).
        If apply=True (default), self.alpha is updated to the new value.
        """
        local_map = local_marker_positions_m if local_marker_positions_m is not None \
            else self.local_marker_positions_m
        if not local_map:
            raise ValueError(
                "No local_marker_positions_m given (neither as an argument nor set at "
                "construction time). Supply {marker_id: (x,y,z)_meters} for the flowbot's "
                "markers in the manipulator/local frame -- see this method's docstring."
            )

        # Collect samples, keyed by marker_id -> list of observed world positions.
        collected: Dict[int, list] = {mid: [] for mid in local_map}
        deadline = time.perf_counter() + timeout_s
        n_taken = 0
        while n_taken < n_samples and time.perf_counter() < deadline:
            markers, _ = self.get_latest_markers()
            for mid in local_map:
                if mid in markers:
                    collected[mid].append(markers[mid])
            n_taken += 1
            time.sleep(sample_interval_s)

        # Average per marker_id; only keep markers seen at least once.
        marker_ids_used = [mid for mid, pts in collected.items() if pts]
        if len(marker_ids_used) < 3:
            raise RuntimeError(
                f"Only {len(marker_ids_used)} of {len(local_map)} known markers were seen "
                f"(need >=3) -- check the rig is in view, labeled_markers is populated by "
                f"this NatNet server, and local_marker_positions_m's keys match the real "
                f"ids get_latest_markers() reports (print it to check)."
            )
        P_local = np.array([local_map[mid] for mid in marker_ids_used], dtype=float)
        P_world = np.array([np.mean(collected[mid], axis=0) for mid in marker_ids_used], dtype=float)

        R_LW, fit_rms_m, per_point_resid_m = _kabsch_rotation(P_local, P_world)  # local -> world
        per_marker_resid_m = {mid: float(r) for mid, r in zip(marker_ids_used, per_point_resid_m)}

        # Known: R_(M<-W) = Rz(alpha) @ R_MW  (opti_to_manip's world->manipulator rotation).
        # Fitted: R_LW = R_(W<-M), so R_(M<-W) = R_LW^T.
        # => Rz(alpha) = R_LW^T @ R_MW^T
        M = R_LW.T @ self.R_MW.T
        alpha_rad = float(np.arctan2(M[1, 0], M[0, 0]))

        # Health check: how far M is from a pure Z-rotation (0 = exact).
        Rz_fit = self.Rz(alpha_rad)
        z_rotation_deviation = float(np.linalg.norm(M - Rz_fit))

        if apply:
            self.alpha = alpha_rad

        result = {
            "alpha_rad": alpha_rad,
            "alpha_deg": float(np.degrees(alpha_rad)),
            "n_markers_used": len(marker_ids_used),
            "marker_ids_used": marker_ids_used,
            "fit_rms_m": fit_rms_m,
            "z_rotation_deviation": z_rotation_deviation,
            "per_marker_resid_m": per_marker_resid_m,
        }
        print(f"[calibrate] alpha = {result['alpha_deg']:.3f} deg "
              f"(markers used: {marker_ids_used}, fit RMS: {fit_rms_m*1000:.2f} mm, "
              f"non-Z-rotation residual: {z_rotation_deviation:.4f})")
        print("[calibrate] per-marker fit residual (mm) -- an outlier here points at "
              "a wrong local position for that specific marker, not general noise:")
        for mid in sorted(per_marker_resid_m, key=lambda m: -per_marker_resid_m[m]):
            print(f"    marker #{mid}: {per_marker_resid_m[mid]*1000:.2f} mm")
        if apply:
            print(f"[calibrate] self.alpha updated to {alpha_rad:.6f} rad")
        return result

    def diagnose_marker_layout(
        self,
        local_marker_positions_m: Optional[Dict[int, Tuple[float, float, float]]] = None,
        n_samples: int = 20,
        sample_interval_s: float = 0.05,
        timeout_s: float = 5.0,
        reference_marker: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Compare the ASSUMED marker layout (local_marker_positions_m) against
        live data WITHOUT needing to know alpha -- useful when
        calibrate_alpha_from_markers's fit_rms_m stays large even after fixing
        an individual marker, since that suggests something systematic (e.g.
        the whole ring's angular handedness assumed backwards) rather than one
        more marker to swap.

        How: average live marker positions, apply only R_MW (not alpha) to
        get each marker's "pre-alpha" (x,y) in the X_local-Y_local plane, and
        compute each marker's angle there. Pick one marker as reference and
        compute every other marker's angle RELATIVE to it -- this relative
        angle is independent of alpha (alpha just rotates everything by a
        common offset, which cancels out in a difference). Compare that
        measured relative angle against the same relative angle computed from
        local_marker_positions_m.

        A consistent handedness/mirror error shows up as every marker's
        (measured - assumed) relative-angle error having the SAME SIGN and
        growing with angular distance from the reference marker (e.g. roughly
        2x the assumed angle, since a sign flip doubles the gap) -- as
        opposed to one single marker being an outlier while the rest are ~0,
        which would mean just that one marker's assumed position is wrong.

        Prints a table and returns {marker_id: (measured - assumed) relative
        angle error in degrees}.
        """
        local_map = local_marker_positions_m if local_marker_positions_m is not None \
            else self.local_marker_positions_m
        if not local_map:
            raise ValueError("No local_marker_positions_m given -- see calibrate_alpha_from_markers.")

        collected: Dict[int, list] = {mid: [] for mid in local_map}
        deadline = time.perf_counter() + timeout_s
        n_taken = 0
        while n_taken < n_samples and time.perf_counter() < deadline:
            markers, _ = self.get_latest_markers()
            for mid in local_map:
                if mid in markers:
                    collected[mid].append(markers[mid])
            n_taken += 1
            time.sleep(sample_interval_s)

        seen = [mid for mid, pts in collected.items() if pts]
        if len(seen) < 3:
            raise RuntimeError(f"Only {len(seen)} of {len(local_map)} known markers were seen (need >=3).")
        world_avg = {mid: np.mean(collected[mid], axis=0) for mid in seen}

        # Pre-alpha local-ish coordinates: R_MW @ (p - centroid), no alpha applied.
        centroid = np.mean(list(world_avg.values()), axis=0)
        pre_alpha = {mid: self.R_MW @ (p - centroid) for mid, p in world_avg.items()}

        ref = reference_marker if reference_marker is not None else seen[0]
        if ref not in seen:
            raise ValueError(f"reference_marker {ref} was not seen (seen: {seen})")

        meas_angle = {mid: float(np.degrees(np.arctan2(pre_alpha[mid][1], pre_alpha[mid][0]))) for mid in seen}
        assumed_angle = {mid: float(np.degrees(np.arctan2(local_map[mid][1], local_map[mid][0]))) for mid in seen}

        def _wrap(a):
            return (a + 180.0) % 360.0 - 180.0

        print(f"[diagnose] reference marker: #{ref}")
        print(f"{'marker':>8} {'meas rel (deg)':>16} {'assumed rel (deg)':>18} {'error (deg)':>12} "
              f"{'meas radius (mm)':>18} {'assumed radius (mm)':>20}")
        errors: Dict[int, float] = {}
        for mid in sorted(seen):
            meas_rel = _wrap(meas_angle[mid] - meas_angle[ref])
            assumed_rel = _wrap(assumed_angle[mid] - assumed_angle[ref])
            err = _wrap(meas_rel - assumed_rel)
            errors[mid] = err
            meas_r = float(np.hypot(pre_alpha[mid][0], pre_alpha[mid][1]) * 1000)
            assumed_r = float(np.hypot(local_map[mid][0], local_map[mid][1]) * 1000)
            print(f"{'#' + str(mid):>8} {meas_rel:16.2f} {assumed_rel:18.2f} {err:12.2f} "
                  f"{meas_r:18.2f} {assumed_r:20.2f}")

        non_ref_errors = [errors[m] for m in seen if m != ref]
        if non_ref_errors and all(abs(e) > 5 for e in non_ref_errors) and \
           (all(e > 0 for e in non_ref_errors) or all(e < 0 for e in non_ref_errors)):
            print("[diagnose] All non-reference errors share the same sign and are all >5deg -- "
                  "this looks like a systematic handedness/mirror mismatch (try negating the "
                  "angle convention in build_ring_marker_positions), not an individual marker swap.")
        return errors

    def transform_to_manip_mm(
        self,
        sample: Optional[RigidBodySample],
        origin: np.ndarray,
        signs=(1.0, 1.0, 1.0),
    ) -> Optional[np.ndarray]:
        """
        Convert a RigidBodySample to manipulator-frame mm with optional axis signs.

        Returns None if sample is None.
        signs: per-axis multiplier, e.g. (-1,-1,1) to flip X and Y.
        """
        if sample is None:
            return None
        t = self.opti_to_manip(np.array(sample.pos_xyz, dtype=float), origin)
        return t * np.array(signs, dtype=float)
        
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', action='store_true',
                        help="Run calibrate_alpha_from_markers() once, print the result, and exit "
                             "-- instead of the default live marker-printing loop.")
    parser.add_argument('--diagnose', action='store_true',
                        help="Run diagnose_marker_layout() once, print the per-marker relative-angle "
                             "comparison, and exit -- use when calibrate's fit_rms_m stays large.")
    parser.add_argument('--server_ip', default="150.65.146.84")
    parser.add_argument('--local_ip', default="150.65.146.84")
    parser.add_argument('--rigid_body_id', type=int, default=1)
    args = parser.parse_args()

    motive = MotiveNatNetReader(
        server_ip=args.server_ip,
        local_ip=args.local_ip,
        use_multicast=False,
        rigid_body_id=args.rigid_body_id,
        local_marker_positions_m=FLOWBOT_MARKER_POSITIONS_M,
    )
    motive.start()

    if args.calibrate:
        try:
            print("\nWaiting for markers to start streaming...")
            time.sleep(2.0)
            n_visible = len(motive.get_latest_markers()[0])
            print(f"{n_visible}/{len(FLOWBOT_MARKER_POSITIONS_M)} markers currently visible.")
            if n_visible < 3:
                print("Fewer than 3 visible -- check the rig is unobstructed before calibrating.")

            result = motive.calibrate_alpha_from_markers()
            print("\nCalibration result:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"\nCalibration failed: {e}")
        finally:
            motive.stop()
        return

    if args.diagnose:
        try:
            print("\nWaiting for markers to start streaming...")
            time.sleep(2.0)
            n_visible = len(motive.get_latest_markers()[0])
            print(f"{n_visible}/{len(FLOWBOT_MARKER_POSITIONS_M)} markers currently visible.")
            if n_visible < 3:
                print("Fewer than 3 visible -- check the rig is unobstructed before diagnosing.")
            motive.diagnose_marker_layout()
        except Exception as e:
            print(f"\nDiagnosis failed: {e}")
        finally:
            motive.stop()
        return

    logger = SyncCSVLogger("teleop_motive_sync.csv")
    try:
        while True:
            t_robot = time.perf_counter()
            motive_sample = motive.get_latest()
            print(motive_sample)
            # Marker index (1-6) + world positions.
            markers, _ = motive.get_latest_markers()
            for mid in sorted(markers):
                x, y, z = markers[mid]
                print(f"  marker #{mid}: ({x:.4f}, {y:.4f}, {z:.4f}) m")
            time.sleep(1)  # simulate some processing delay
            # logger.log(t_robot=t_robot, pc_xyz=pc, pwm_123=pwm, motive=motive_sample)
    finally:
        motive.stop()
        # logger.close()

if __name__ == "__main__":
    main()
