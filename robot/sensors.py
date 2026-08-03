"""
Sensor models for the differential-drive robot: healthy behaviour only.

Takes a noiseless trajectory from `trajectories.rollout` and produces what
the vehicle's sensors would actually report. Faults are deliberately absent
-- they belong in a separate module, so that "what a healthy sensor does" and
"what a broken one does" never become entangled in one place.

MEASUREMENT SUITE

    encoder_left    left wheel angular rate    (rad/s)
    encoder_right   right wheel angular rate   (rad/s)
    gyro_z          yaw rate                   (rad/s)

Three measurements for two underlying quantities (forward speed and turn
rate). That redundancy is intentional. With an exactly determined suite, a
corrupted channel simply produces a different, equally plausible state
estimate and nothing in the data reveals the fault. With one spare degree of
freedom the channels can be checked against each other, and a fault appears
as inconsistency rather than as a shifted answer. The encoders and the gyro
observe overlapping information -- both constrain turn rate -- which is what
makes cross-channel disagreement meaningful.

NOISE IS REGIME-DEPENDENT, AND THIS IS THE POINT

Sensor noise here is not a constant. Encoder noise grows with wheel speed
(vibration and slip increase with rotation rate) and gyro noise grows with
turn rate. This is the structure a heteroscedastic model exists to learn: if
noise were constant, a single tuned scalar would be optimal and there would
be nothing for a learned noise model to contribute. The relationships are
smooth functions of quantities that are themselves observable, so they are
learnable in principle -- which makes any failure to learn them attributable
to the method rather than to the data.

QUANTISATION

A real encoder counts discrete ticks, so its output is quantised rather than
continuous. Rate is recovered by differencing counts over an interval, which
turns the position quantum into a rate quantum of `resolution / dt`. This
noise is bounded and uniform rather than Gaussian, and it dominates at low
speeds where the Gaussian term is small. It is included because it is what
distinguishes an encoder from a generic noisy sensor, and because a
non-Gaussian noise floor is a more honest test of a model that assumes
Gaussian likelihoods.

Run: python sensors.py
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("encoder_left", "encoder_right", "gyro_z")
N_CHANNELS = len(CHANNELS)

# --- noise specification --------------------------------------------------
ENC_SIGMA_BASE = 0.08         # rad/s, encoder noise floor
ENC_SIGMA_GAIN = 0.05         # extra noise per rad/s of wheel rate
GYRO_SIGMA_BASE = 0.010       # rad/s, gyro noise floor
GYRO_SIGMA_GAIN = 0.030       # extra noise per rad/s of turn rate
GYRO_BIAS_STD = 0.005         # rad/s, fixed offset drawn once per episode

ENCODER_TICKS_PER_REV = 1024  # quantisation of the wheel encoders


def encoder_sigma(wheel_rate):
    """Encoder noise standard deviation, rad/s. Grows with wheel speed."""
    return ENC_SIGMA_BASE * (1.0 + ENC_SIGMA_GAIN * np.abs(wheel_rate))


def gyro_sigma(omega):
    """Gyro noise standard deviation, rad/s. Grows with turn rate."""
    return GYRO_SIGMA_BASE * (1.0 + GYRO_SIGMA_GAIN * np.abs(omega) / 0.01)


def _quantise_rate(rate, dt, ticks_per_rev=ENCODER_TICKS_PER_REV):
    """Quantise an angular rate as a tick-counting encoder would.

    Counts are integers, so a rate measured by differencing counts over `dt`
    can only take multiples of (2*pi / ticks_per_rev) / dt.
    """
    quantum = (2.0 * np.pi / ticks_per_rev) / dt
    return np.round(rate / quantum) * quantum


def measure(traj, seed: int, dt: float | None = None, quantise: bool = True):
    """Simulate healthy sensor readings for a trajectory.

    Parameters
    ----------
    traj : dict from trajectories.rollout
    seed : per-episode RNG seed; also fixes the gyro bias for this episode
    dt   : sample interval, needed for encoder quantisation
    quantise : include encoder tick quantisation

    Returns a dict with one array per channel, plus the per-sample noise
    standard deviations that generated them. Those sigmas are ground truth
    for the noise level -- the quantity a heteroscedastic model is trying to
    recover -- and are kept so that a learned estimate can be scored against
    the value that actually produced the data.
    """
    rng = np.random.default_rng(seed)
    if dt is None:
        dt = float(traj["t"][1] - traj["t"][0])

    rate_l, rate_r, omega = traj["rate_left"], traj["rate_right"], traj["omega"]

    sig_l = encoder_sigma(rate_l)
    sig_r = encoder_sigma(rate_r)
    sig_g = gyro_sigma(omega)

    z_l = rate_l + rng.normal(0.0, 1.0, len(rate_l)) * sig_l
    z_r = rate_r + rng.normal(0.0, 1.0, len(rate_r)) * sig_r

    # a gyro's bias is fixed within a run but differs between units/power
    # cycles; drawing it once per episode keeps it a property of the episode
    # rather than of the timestep
    bias = rng.normal(0.0, GYRO_BIAS_STD)
    z_g = omega + bias + rng.normal(0.0, 1.0, len(omega)) * sig_g

    if quantise:
        z_l = _quantise_rate(z_l, dt)
        z_r = _quantise_rate(z_r, dt)

    return {
        "encoder_left": z_l, "encoder_right": z_r, "gyro_z": z_g,
        "sigma_encoder_left": sig_l, "sigma_encoder_right": sig_r,
        "sigma_gyro_z": sig_g, "gyro_bias": bias,
    }


def stack(meas):
    """Channels as an (N, 3) array in CHANNELS order."""
    return np.column_stack([meas[c] for c in CHANNELS])


def _self_test():
    from trajectories import DT, random_trajectory, s_path
    ok = True

    traj = s_path()
    m = measure(traj, seed=0)
    print(f"channels: {CHANNELS}, {len(m['encoder_left'])} samples")

    # --- noise level matches the specification -------------------------
    # Two independent noise sources are present, so the expected spread is
    # their quadrature sum. Checking against the Gaussian term alone would
    # under-predict by the quantisation contribution, which for a uniform
    # quantum q has standard deviation q/sqrt(12).
    quantum = (2.0 * np.pi / ENCODER_TICKS_PER_REV) / DT
    m_raw = measure(traj, seed=0, quantise=False)
    res_gauss = m_raw["encoder_left"] - traj["rate_left"]
    emp_g, spec_g = float(res_gauss.std()), float(m_raw["sigma_encoder_left"].mean())
    g_ok = abs(emp_g - spec_g) / spec_g < 0.15
    print(f"gaussian term   empirical {emp_g:.4f} vs specified {spec_g:.4f}  "
          f"{'ok' if g_ok else 'FAIL'}")

    res_l = m["encoder_left"] - traj["rate_left"]
    emp_tot = float(res_l.std())
    spec_tot = float(np.sqrt(spec_g ** 2 + quantum ** 2 / 12.0))
    t_ok = abs(emp_tot - spec_tot) / spec_tot < 0.15
    print(f"gaussian+quant  empirical {emp_tot:.4f} vs predicted {spec_tot:.4f}  "
          f"{'ok' if t_ok else 'FAIL'}")
    print(f"   (quantisation contributes {quantum/np.sqrt(12):.4f} rad/s -- "
          f"comparable to the Gaussian floor, so it is not negligible)")
    ok &= g_ok and t_ok

    # --- noise really is regime-dependent ------------------------------
    # split by wheel speed and confirm the spread grows
    fast = np.abs(traj["rate_left"]) > np.percentile(np.abs(traj["rate_left"]), 75)
    slow = np.abs(traj["rate_left"]) < np.percentile(np.abs(traj["rate_left"]), 25)
    s_fast, s_slow = float(res_l[fast].std()), float(res_l[slow].std())
    het_ok = s_fast > s_slow
    print(f"heteroscedastic  slow {s_slow:.4f} < fast {s_fast:.4f}  "
          f"{'ok' if het_ok else 'FAIL'}")
    ok &= het_ok

    # --- quantisation grid is respected --------------------------------
    quantum = (2.0 * np.pi / ENCODER_TICKS_PER_REV) / DT
    on_grid = np.allclose(m["encoder_left"] / quantum,
                          np.round(m["encoder_left"] / quantum))
    print(f"encoder on tick grid  (quantum {quantum:.4f} rad/s)  "
          f"{'ok' if on_grid else 'FAIL'}")
    ok &= on_grid

    # --- measurements remain informative about the commands ------------
    # invert the encoder pair and compare against truth
    from dynamics import WHEEL_RADIUS, commands_from_wheel_speeds
    v_hat, w_hat = commands_from_wheel_speeds(
        m["encoder_left"] * WHEEL_RADIUS, m["encoder_right"] * WHEEL_RADIUS)
    e_v = float(np.sqrt(np.mean((v_hat - traj["v"]) ** 2)))
    e_w = float(np.sqrt(np.mean((w_hat - traj["omega"]) ** 2)))
    inv_ok = e_v < 0.05 and e_w < 0.15
    print(f"encoder inversion  v err {e_v:.4f} m/s, omega err {e_w:.4f} rad/s  "
          f"{'ok' if inv_ok else 'FAIL'}")
    ok &= inv_ok

    # --- redundancy: gyro and encoders agree on turn rate --------------
    agree = float(np.corrcoef(w_hat, m["gyro_z"])[0, 1])
    red_ok = agree > 0.95
    print(f"encoder/gyro agreement on omega  corr {agree:.4f}  "
          f"{'ok' if red_ok else 'FAIL'}")
    ok &= red_ok
    print("   (this redundancy is what makes a single-channel fault visible)")

    # --- reproducibility ------------------------------------------------
    same = np.allclose(measure(traj, seed=0)["gyro_z"], m["gyro_z"])
    diff = not np.allclose(measure(traj, seed=1)["gyro_z"], m["gyro_z"])
    print(f"seed reproducible / distinct  {'ok' if same and diff else 'FAIL'}")
    ok &= same and diff

    # --- gyro bias is per-episode, not per-sample -----------------------
    biases = [measure(random_trajectory(s), seed=s)["gyro_bias"]
              for s in range(200)]
    b_std = float(np.std(biases))
    bias_ok = abs(b_std - GYRO_BIAS_STD) / GYRO_BIAS_STD < 0.25
    print(f"gyro bias spread across episodes  {b_std:.5f} vs "
          f"{GYRO_BIAS_STD:.5f}  {'ok' if bias_ok else 'FAIL'}")
    ok &= bias_ok

    # --- quantisation dominates at low speed ---------------------------
    slow_traj = random_trajectory(3)
    mq = measure(slow_traj, seed=3, quantise=True)
    mc = measure(slow_traj, seed=3, quantise=False)
    delta = float(np.abs(mq["encoder_left"] - mc["encoder_left"]).mean())
    print(f"quantisation shifts readings by {delta:.4f} rad/s on average "
          f"(quantum {quantum:.4f})")

    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    _self_test()
