
from typing import Callable, Optional, Tuple, Dict
import numpy as np

ArrayLike = np.ndarray

class Flow_driven_bellow:
    """
    Parameters for one manipulator module.

    Required geometric params (paper):
      - Aeff: effective cross sectional area of one bellow (Eq. 1, 22)
      - l0: initial active length of bellow (Eq. 2, 21, 23)
      - d: distance from center axis to each bellow center (Eq. 7, 17)
      - lb: lower passive structure length (Eq. 14)
      - lu: upper passive structure length (Eq. 10 to 12, 18 to 20)

    Adjustment function self:
      - a0, a1 for f(p0) = a1 * p0 + a0 (Eq. 3)

    Stiffness model:
      - k_model: either a constant scalar, or a callable k = k_model(delta_l)
        where delta_l is the axial deformation (same unit as length).
    """
    def __init__(self,
                D_in: float,   # Diameter of inner section of the bellow (small section)
                D_out: float,  # Diameter of outer section of the bellow (big section)
                l0: float,
                d: float,
                lb: float,
                lu: float,
                k_model: float | Callable[[float], float] = 1.0,  # e.g. lambda dl: 1000.0 + 200.0 * dl
                a_delta : float = 0.0,
                b_delta : float = 0.0,
                a_pwm2press: float = 0.0,
                b_pwm2press: float = 0.0,
            ):
        self.D_in = D_in
        self.D_out = D_out
        self.Aeff = np.pi * (D_in + D_out)**2/(4*4)
        self.l0 = l0
        self.d = d
        self.lb = lb
        self.lu = lu
        self.k_model = k_model
        self.a_delta = a_delta
        self.b_delta = b_delta
        self.a_pwm2press = a_pwm2press
        self.b_pwm2press = b_pwm2press

    def _clip_pressure(self, pb: np.ndarray) -> np.ndarray:
        return np.maximum(pb, 0.0)

    def val_from_model(self,k_model: float | Callable[[float], float], x: float) -> float:
        if callable(k_model):
            val = float(k_model(float(x)))
            if not np.isfinite(val):
                raise ValueError(f"k_model(delta_l) returned invalid stiffness: {val}")
            return val
        val = float(k_model)
        if not np.isfinite(val):
            raise ValueError(f"Constant stiffness k_model must be > 0, got: {val}")
        return val

    def pwm_to_pressure(self,
                        pwm)-> np.ndarray:
        pwm = np.asarray(pwm, dtype=int).reshape(3,)
        pressure = np.zeros(3, dtype=float)
        for i in range(3):
            if pwm[i] > 0:
                pressure[i] = pwm[i] * self.a_pwm2press + self.b_pwm2press
            else:
                pressure[i] = 0.0
        return pressure
    
    def pressure_to_pwm(self,
                        p,
                        pwm_max = 30)-> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(3,)
        pwm = np.zeros(3, dtype=np.int32)
        for i in range(3):
            pwm[i] = (p[i] - self.b_pwm2press)/self.a_pwm2press
            if pwm[i] < 0:
                pwm[i] = 0
                
        pwm = np.clip(pwm, 0, pwm_max)
        return pwm

    def pressures_to_lengths(self,
        pb: ArrayLike,
    ) -> np.ndarray:
        """
        Map gauge pressures pb (shape (3,)) to bellow arc lengths l_i (shape (3,)).
        Implements Eq. (1) to (4) and Eq. (2).

        Notes:
        - We implement p0_i as pb_i - sgn(pb_i)*min_abs, where min_abs = min(|pb|).
        - f(p0) = a1*p0 + a0.
        """
        pb = np.asarray(pb, dtype=float).reshape(3,)
        min_abs = float(np.min(np.abs(pb)))

        p0 = np.zeros(3, dtype=float)
        for i in range(3):
            s = np.sign(pb[i])
            if s == 0:
                p0[i] = 0.0
            else:
                p0[i] = pb[i] - s * min_abs

        # Eq. (1) and (2): delta_l = pb*Aeff/k, l = delta_l + l0 + f(p0)
        l = np.zeros(3, dtype=float)
        for i in range(3):
            delta_l = 0.0
            for _ in range(5):
                k = self.val_from_model(self.k_model, delta_l)
                delta_l_new = (pb[i] * self.Aeff) / k
                if abs(delta_l_new - delta_l) < 1e-9:
                    delta_l = delta_l_new
                    break
                delta_l = delta_l_new

            f = self.a_delta * p0[i] + self.b_delta
            l[i] = delta_l + self.l0 + f

        return l


    def lengths_to_config(self,
        l: ArrayLike,
    ) -> Tuple[float, float, float, float]:
        """
        From bellow lengths l1,l2,l3 -> (lc, phi, kappa, theta)
        Implements Eq. (5) to (8).

        We use a standard atan2 form for phi that matches the 3 actuator geometry:
        phi = atan2( sqrt(3)*(l2 - l3), (2*l1 - l2 - l3) )

        kappa uses Eq. (7) as written in the paper.
        """
        l = np.asarray(l, dtype=float).reshape(3,)
        l1, l2, l3 = float(l[0]), float(l[1]), float(l[2])

        lc = (l1 + l2 + l3) / 3.0

        num = np.sqrt(3.0) * (l2 - l3)
        den = (2.0 * l1 - l2 - l3)
        phi = float(np.arctan2(num, den))

        # Eq. (7)
        s = l1*l1 + l2*l2 + l3*l3 - l1*l2 - l1*l3 - l2*l3
        s = max(0.0, float(s))
        kappa = (2.0 * np.sqrt(s)) / (self.d * (l1 + l2 + l3))

        theta = lc * kappa
        return lc, phi, kappa, theta


    def forward_kinematics_from_lengths(self,
        l: ArrayLike,
    ) -> Dict[str, np.ndarray]:
        """
        Forward kinematics:
        - returns pc (top center) using Eq. (9) to (12)
        - also returns (lc, phi, kappa, theta)

        Output dict keys:
        - "pc": np.array([xc, yc, zc])
        - "lc", "phi", "kappa", "theta"
        """
        lc, phi, kappa, theta = self.lengths_to_config(l)

        # Handle near straight case kappa ~ 0 using limits
        eps = 1e-9
        if abs(kappa) < eps:
            # straight: x=y=0, z increases by lc plus passive segments
            pc = np.array([0.0, 0.0, self.lb + lc + self.lu], dtype=float)
            return {
                "pc": pc,
                "lc": np.array([lc]),
                "phi": np.array([phi]),
                "kappa": np.array([kappa]),
                "theta": np.array([theta]),
            }

        rho = 1.0 / kappa  # radius of curvature
        # Eq. (9)
        c = 2.0 * rho * np.sin(theta / 2.0)

        # Eq. (10) to (12)
        xc = c * np.sin(theta / 2.0) * np.cos(phi) + self.lu * np.sin(theta) * np.cos(phi)
        yc = c * np.sin(theta / 2.0) * np.sin(phi) + self.lu * np.sin(theta) * np.sin(phi)
        zc = c * np.cos(theta / 2.0) + self.lb + self.lu * np.cos(theta)

        pc = np.array([xc, yc, zc], dtype=float)
        return {
            "pc": pc,
            "lc": np.array([lc]),
            "phi": np.array([phi]),
            "kappa": np.array([kappa]),
            "theta": np.array([theta]),
        }


    def forward_kinematics_from_pressures(self,
        pb: ArrayLike,
    ) -> Dict[str, np.ndarray]:
        """
        pb -> l -> pc
        """
        l = self.pressures_to_lengths(pb)
        out = self.forward_kinematics_from_lengths(l)
        out["l"] = np.asarray(l, dtype=float)
        return out


    def inverse_kinematics_position_to_lengths(self,
        pc: ArrayLike,
        eps: float = 1e-9,
        strict: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Inverse kinematics from task point pc (OptiTrack point) to bellow lengths.
        Assumptions:
        - pc is the tip center point in the base frame (the point after the upper passive link lu)
        - constant curvature model, single module
        - lb is the base passive offset along base z
        - lu is the upper passive (rigid) link length

        Returns:
        - l: (3,) bellow lengths
        - phi, theta, kappa, lc
        - p: (3,) intermediate point (end of active section before adding lu)
        - t: (3,) tangent direction at the end of the active section
        """
        pc = np.asarray(pc, dtype=float).reshape(3,)
        xc, yc, zc = float(pc[0]), float(pc[1]), float(pc[2])

        r = float(np.hypot(xc, yc))
        phi = float(np.arctan2(yc, xc)) if r > eps else 0.0

        s = zc - self.lb  # z after removing base offset

        denom = s + self.lu
        if denom <= eps:
            msg = f"Invalid geometry: (zc - lb + lu) must be > 0, got {denom}."
            if strict:
                raise ValueError(msg)
            # fallback: clamp to avoid blow up
            denom = eps

        # Closed form: tan(theta/2) = r / (s + lu)
        t_half = r / denom
        theta = float(2.0 * np.arctan(t_half))

        # Straight or near straight case
        if abs(theta) < 1e-8 or r < eps:
            kappa = 0.0
            # For straight: zc = lb + lc + lu
            lc = float(zc - self.lb - self.lu)
            if lc < 0 and strict:
                raise ValueError(f"Computed lc < 0 in straight case: {lc}")
            lc = max(0.0, lc)

            l = np.array([lc, lc, lc], dtype=float)

            t_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            p = pc - self.lu * t_vec

            return {
                "l": l,
                "phi": np.array([phi]),
                "theta": np.array([theta]),
                "kappa": np.array([kappa]),
                "lc": np.array([lc]),
                "p": p,
                "t": t_vec,
            }

        sin_th = float(np.sin(theta))
        cos_th = float(np.cos(theta))

        if abs(sin_th) < eps:
            msg = "sin(theta) too small for stable rho computation."
            if strict:
                raise ValueError(msg)
            sin_th = np.sign(sin_th) * eps if sin_th != 0 else eps

        # rho = (s - lu*cos(theta)) / sin(theta)
        rho = (s - self.lu * cos_th) / sin_th
        if rho <= 0 and strict:
            raise ValueError(f"Computed rho <= 0 (invalid for this model): rho={rho}")

        # Use positive rho for curvature magnitude, bending direction is handled by phi
        rho = abs(rho)
        kappa = 1.0 / rho
        lc = rho * theta

        # Tangent direction at tip
        t_vec = np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)], dtype=float)

        # Intermediate point p (before adding lu)
        p = pc - self.lu * t_vec

        # Bellow lengths
        l = np.zeros(3, dtype=float)
        for i in range(3):
            gamma = (2.0 * np.pi / 3.0) * i   # actuator angles: 0, 120, 240 deg
            l[i] = lc + theta * self.d * np.cos(gamma - phi)


        return {
            "l": l,
            "phi": np.array([phi]),
            "theta": np.array([theta]),
            "kappa": np.array([kappa]),
            "lc": np.array([lc]),
            "p": p,
            "t": t_vec,
        }


    def inverse_pressures_from_lengths(self,
        l: ArrayLike,
    ) -> np.ndarray:
        """
        Compute pressures pb from bellow lengths l_i.
        Implements Eq. (21) to (23) with a practical closed form assumption:

        - Choose the base bellow whose p0 is set to 0 (paper logic).
        - For base: delta_l = li - l0 - a0  (Eq. 23)
        - pb_base = delta_l * k / Aeff (Eq. 22)

        For other bellows, assume sign(pb_i) follows sign(delta_l_i),
        and use:
            p0_i = pb_i - sign_i*abs(pb_base)
            delta_l_i = li - l0 - a1*p0_i - a0
            pb_i = delta_l_i * k / Aeff
        Solve for pb_i in closed form when k is treated locally constant.

        If your k_model strongly depends on delta_l, you can increase iterations.
        """
        l = np.asarray(l, dtype=float).reshape(3,)

        # Decide whether expansion or contraction dominates relative to l0
        # Following the paper: if li > l0, base is the shortest l;
        # if li < l0, base is the longest l.
        if np.mean(l) >= self.l0:
            base_idx = int(np.argmin(l))
        else:
            base_idx = int(np.argmax(l))

        pb = np.zeros(3, dtype=float)

        # Base bellow (p0 = 0)
        delta_l_base = float(l[base_idx] - self.l0 - self.b_delta)
        if delta_l_base < 0:
            delta_l_base = 0.0
        k_base = self.val_from_model(self.k_model, delta_l_base)
        pb_base = (delta_l_base * k_base) / self.Aeff
        pb[base_idx] = pb_base

        min_abs = abs(pb_base)

        # Others
        for i in range(3):
            if i == base_idx:
                continue
            # initial guess for delta_l and k
            delta_l_i = float(l[i] - self.l0 - self.b_delta)
            sign_i = 1.0 if delta_l_i >= 0 else -1.0

            # fixed point refine for k(delta_l)
            pb_i = 0.0
            for _ in range(8):
                k_i = self.val_from_model(self.k_model, delta_l_i)
                alpha = k_i / self.Aeff

                # Closed form:
                # pb_i * (1 + alpha*a1) = alpha*(li - l0 - a0 + a1*sign_i*min_abs)
                rhs = alpha * (float(l[i] - self.l0 - self.b_delta) + self.a_delta * sign_i * min_abs)
                denom = (1.0 + alpha * self.a_delta)
                pb_i_new = rhs / denom
                if pb_i_new < 0:
                    pb_i_new = 0.0
                # update delta_l using Eq. (21)
                p0_i = pb_i_new - sign_i * min_abs
                delta_l_i_new = float(l[i] - self.l0 - self.b_delta - self.a_delta * p0_i)

                if abs(pb_i_new - pb_i) < 1e-9:
                    pb_i = pb_i_new
                    delta_l_i = delta_l_i_new
                    break

                pb_i = pb_i_new
                delta_l_i = delta_l_i_new

            pb[i] = pb_i
        pb = self._clip_pressure(pb) 
        return pb


    def inverse_pressures_from_position(self,
        p: ArrayLike,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience: p -> lengths -> pressures.
        """
        ik = self.inverse_kinematics_position_to_lengths(p)
        pb = self.inverse_pressures_from_lengths(ik["l"])
        pwm = self.pressure_to_pwm(pb)
        ik["pb"] = pb
        ik["pwm"] = pwm
        return ik


# Optional: example stiffness model placeholder
def example_k_piecewise(delta_l: float) -> float:
    """
    Placeholder for a stiffness law like Eq. (24) in the paper.
    You should replace this with your own model and units.

    delta_l here is "deformed length" in your chosen unit.
    """
    # Example only: return a positive stiffness
    return 1.0


if __name__ == "__main__":
    # Example usage (fill parameters later)
    model = Flow_driven_bellow(
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
    pwm_signals = np.array([10, 5, 20], dtype=int)
    pb = model.pwm_to_pressure(pwm=pwm_signals)
    print(f"pressure = {pb}")
    fk = model.forward_kinematics_from_pressures(pb)
    print("Forward pc:", fk["pc"], "lengths:", fk["l"])

    p = np.array([-10.684178252962187 ,-23.44879952771703 ,87.10544837767496], dtype=float)

    # ik = model.inverse_kinematics_position_to_lengths(p)
    # print(ik)
    ik = model.inverse_pressures_from_position(p)
    print("Inverse lengths:", ik["l"], "pressures:", ik["pb"], "pwm:", ik["pwm"])
