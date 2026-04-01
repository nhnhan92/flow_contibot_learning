"""
Flow_driven_bellow (linear PWM-to-pressure variant).

Uses a simple linear model:
    pressure = a_pwm2press * pwm + b_pwm2press
    pwm      = (pressure - b_pwm2press) / a_pwm2press

Identical to kinematic_modeling.py in every other respect.
Default linear params (from calibration): a=0.004227, b=0.012059
"""

from typing import Callable, Optional, Tuple, Dict
import numpy as np

ArrayLike = np.ndarray


class Flow_driven_bellow_linear:
    """
    Kinematic model using a linear PWM <-> pressure mapping.

    pwm_to_pressure : pressure = a * pwm + b
    pressure_to_pwm : pwm = (pressure - b) / a   (clamped to [0, pwm_max])
    """
    def __init__(self,
                D_in: float,
                D_out: float,
                l0: float,
                d: float,
                lb: float,
                lu: float,
                k_model: float | Callable[[float], float] = 1.0,
                a_delta: float = 0.0,
                b_delta: float = 0.0,
                a_pwm2press: float = 0.004227,
                b_pwm2press: float = 0.012059,
            ):
        self.D_in  = D_in
        self.D_out = D_out
        self.Aeff  = np.pi * (D_in + D_out)**2 / (4 * 4)
        self.l0    = l0
        self.d     = d
        self.lb    = lb
        self.lu    = lu
        self.k_model = k_model
        self.a_delta = a_delta
        self.b_delta = b_delta
        self.a_pwm2press = a_pwm2press
        self.b_pwm2press = b_pwm2press

    def _clip_pressure(self, pb: np.ndarray) -> np.ndarray:
        return np.maximum(pb, 0.0)

    def val_from_model(self, k_model: float | Callable[[float], float], x: float) -> float:
        if callable(k_model):
            val = float(k_model(float(x)))
            if not np.isfinite(val):
                raise ValueError(f"k_model(delta_l) returned invalid stiffness: {val}")
            return val
        val = float(k_model)
        if not np.isfinite(val):
            raise ValueError(f"Constant stiffness k_model must be > 0, got: {val}")
        return val

    # ------------------------------------------------------------------
    # Linear PWM <-> pressure
    # ------------------------------------------------------------------
    def pwm_to_pressure(self, pwm) -> np.ndarray:
        """pressure = a * pwm + b  (per bellow)"""
        pwm = np.asarray(pwm, dtype=float).reshape(3,)
        pressure = np.where(pwm > 0,
                            self.a_pwm2press * pwm + self.b_pwm2press,
                            0.0)
        return np.maximum(pressure, 0.0)

    def pressure_to_pwm(self, p, pwm_max: int = 30) -> np.ndarray:
        """pwm = (pressure - b) / a  (per bellow), clamped to [0, pwm_max]"""
        p = np.asarray(p, dtype=float).reshape(3,)
        pwm = np.where(p > 0,
                       (p - self.b_pwm2press) / self.a_pwm2press,
                       0.0)
        pwm = np.clip(np.round(pwm), 0, pwm_max).astype(np.int32)
        return pwm

    # ------------------------------------------------------------------
    # Everything below is identical to kinematic_modeling.py
    # ------------------------------------------------------------------
    def pressures_to_lengths(self, pb: ArrayLike) -> np.ndarray:
        pb = np.asarray(pb, dtype=float).reshape(3,)
        min_abs = float(np.min(np.abs(pb)))

        p0 = np.zeros(3, dtype=float)
        for i in range(3):
            s = np.sign(pb[i])
            if s == 0:
                p0[i] = 0.0
            else:
                p0[i] = pb[i] - s * min_abs

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

    def lengths_to_config(self, l: ArrayLike) -> Tuple[float, float, float, float]:
        l = np.asarray(l, dtype=float).reshape(3,)
        l1, l2, l3 = float(l[0]), float(l[1]), float(l[2])

        lc = (l1 + l2 + l3) / 3.0

        num = np.sqrt(3.0) * (l2 - l3)
        den = (2.0 * l1 - l2 - l3)
        phi = float(np.arctan2(num, den))

        s = l1*l1 + l2*l2 + l3*l3 - l1*l2 - l1*l3 - l2*l3
        s = max(0.0, float(s))
        kappa = (2.0 * np.sqrt(s)) / (self.d * (l1 + l2 + l3))

        theta = lc * kappa
        return lc, phi, kappa, theta

    def forward_kinematics_from_lengths(self, l: ArrayLike) -> Dict[str, np.ndarray]:
        lc, phi, kappa, theta = self.lengths_to_config(l)

        eps = 1e-9
        if abs(kappa) < eps:
            pc = np.array([0.0, 0.0, self.lb + lc + self.lu], dtype=float)
            return {"pc": pc, "lc": np.array([lc]), "phi": np.array([phi]),
                    "kappa": np.array([kappa]), "theta": np.array([theta])}

        rho = 1.0 / kappa
        c = 2.0 * rho * np.sin(theta / 2.0)

        xc = c * np.sin(theta / 2.0) * np.cos(phi) + self.lu * np.sin(theta) * np.cos(phi)
        yc = c * np.sin(theta / 2.0) * np.sin(phi) + self.lu * np.sin(theta) * np.sin(phi)
        zc = c * np.cos(theta / 2.0) + self.lb + self.lu * np.cos(theta)

        pc = np.array([xc, yc, zc], dtype=float)
        return {"pc": pc, "lc": np.array([lc]), "phi": np.array([phi]),
                "kappa": np.array([kappa]), "theta": np.array([theta])}

    def forward_kinematics_from_pressures(self, pb: ArrayLike) -> Dict[str, np.ndarray]:
        l = self.pressures_to_lengths(pb)
        out = self.forward_kinematics_from_lengths(l)
        out["l"] = np.asarray(l, dtype=float)
        return out

    def inverse_kinematics_position_to_lengths(
        self, pc: ArrayLike, eps: float = 1e-9, strict: bool = False,
    ) -> Dict[str, np.ndarray]:
        pc = np.asarray(pc, dtype=float).reshape(3,)
        xc, yc, zc = float(pc[0]), float(pc[1]), float(pc[2])

        r = float(np.hypot(xc, yc))
        phi = float(np.arctan2(yc, xc)) if r > eps else 0.0

        s = zc - self.lb
        denom = s + self.lu
        if denom <= eps:
            msg = f"Invalid geometry: (zc - lb + lu) must be > 0, got {denom}."
            if strict:
                raise ValueError(msg)
            denom = eps

        t_half = r / denom
        theta = float(2.0 * np.arctan(t_half))

        if abs(theta) < 1e-8 or r < eps:
            kappa = 0.0
            lc = float(zc - self.lb - self.lu)
            if lc < 0 and strict:
                raise ValueError(f"Computed lc < 0 in straight case: {lc}")
            lc = max(0.0, lc)
            l = np.array([lc, lc, lc], dtype=float)
            t_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            p = pc - self.lu * t_vec
            return {"l": l, "phi": np.array([phi]), "theta": np.array([theta]),
                    "kappa": np.array([kappa]), "lc": np.array([lc]), "p": p, "t": t_vec}

        sin_th = float(np.sin(theta))
        cos_th = float(np.cos(theta))

        if abs(sin_th) < eps:
            msg = "sin(theta) too small for stable rho computation."
            if strict:
                raise ValueError(msg)
            sin_th = np.sign(sin_th) * eps if sin_th != 0 else eps

        rho = (s - self.lu * cos_th) / sin_th
        if rho <= 0 and strict:
            raise ValueError(f"Computed rho <= 0: rho={rho}")

        rho = abs(rho)
        kappa = 1.0 / rho
        lc = rho * theta

        t_vec = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)], dtype=float)
        p = pc - self.lu * t_vec

        l = np.zeros(3, dtype=float)
        for i in range(3):
            gamma = (2.0 * np.pi / 3.0) * i
            l[i] = lc + theta * self.d * np.cos(gamma - phi)

        return {"l": l, "phi": np.array([phi]), "theta": np.array([theta]),
                "kappa": np.array([kappa]), "lc": np.array([lc]), "p": p, "t": t_vec}

    def inverse_pressures_from_lengths(self, l: ArrayLike) -> np.ndarray:
        l = np.asarray(l, dtype=float).reshape(3,)

        if np.mean(l) >= self.l0:
            base_idx = int(np.argmin(l))
        else:
            base_idx = int(np.argmax(l))

        pb = np.zeros(3, dtype=float)

        delta_l_base = float(l[base_idx] - self.l0 - self.b_delta)
        if delta_l_base < 0:
            delta_l_base = 0.0
        k_base = self.val_from_model(self.k_model, delta_l_base)
        pb_base = (delta_l_base * k_base) / self.Aeff
        pb[base_idx] = pb_base

        min_abs = abs(pb_base)

        for i in range(3):
            if i == base_idx:
                continue
            delta_l_i = float(l[i] - self.l0 - self.b_delta)
            sign_i = 1.0 if delta_l_i >= 0 else -1.0

            pb_i = 0.0
            for _ in range(8):
                k_i = self.val_from_model(self.k_model, delta_l_i)
                alpha = k_i / self.Aeff

                rhs = alpha * (float(l[i] - self.l0 - self.b_delta) + self.a_delta * sign_i * min_abs)
                denom = (1.0 + alpha * self.a_delta)
                pb_i_new = rhs / denom
                if pb_i_new < 0:
                    pb_i_new = 0.0
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

    def inverse_pressures_from_position(self, p: ArrayLike) -> Dict[str, np.ndarray]:
        """Convenience: position -> lengths -> pressures -> pwm."""
        ik = self.inverse_kinematics_position_to_lengths(p)
        pb  = self.inverse_pressures_from_lengths(ik["l"])
        pwm = self.pressure_to_pwm(pb)
        ik["pb"]  = pb
        ik["pwm"] = pwm
        return ik
