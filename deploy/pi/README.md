# Running the estimator on a Raspberry Pi 5 with a CubeOrange

Status: the software has been run against a simulated vehicle over a UDP
loopback and against replayed simulation data. It has not been run against a
CubeOrange. Everything below about wiring and ArduPilot parameters is from
the ArduPilot documentation for companion computers and has not been
confirmed on this hardware. Confirm each step before enabling the service.

## What runs

One Python process, `deploy/pi_main.py`. It reads `SCALED_IMU` from the
flight controller over MAVLink, steps the estimator in `deploy/runtime.py`,
and once a second sends back as `NAMED_VALUE_FLOAT` the six health
estimates, the filter's NIS, and the model's novelty per device -- how
unlike its training data the model's inputs are, 1 when they are ordinary.
It raises a `STATUSTEXT` alert when NIS or a device's novelty crosses its
level or the estimate stops being usable, and writes one log row per step to
`deploy/logs/` — including the raw reading, so a flight can be replayed
through the estimator afterwards with faults injected into real data.

The estimator is advisory. Nothing it sends feeds the flight controller's
EKF; the vehicle flies on its own estimate whether this process is running
or not. That is the configuration to fly first. The one exception is
`--allow-rtl`, off by default, which is described under *When every sensor
fails* below and should stay off until it has been tried in SITL.

It starts its filter from the flight controller's own attitude, read from
one `ATTITUDE` message before the first update, and prints the roll, pitch
and yaw it started from. If none arrives within two seconds it warns and
starts level at heading 0 -- and in simulation a filter started half a turn
wrong in heading carries the error in its health estimates for the whole
flight (`quad_sim/health_readout.py`), so treat that warning as a reason to
restart, not to fly.

Measured on a laptop: about 2 ms per step for the quadcopter with the
epistemic term (1.5 ms without) against a 20 ms budget at 50 Hz, and 28 MB
resident with the weights exported to numpy. The Pi 5 will be slower;
measure it there with `deploy/runtime.py` before trusting the margin.

## Software on the Pi

    sudo apt install python3-numpy python3-pandas python3-venv
    python3 -m venv --system-site-packages ~/ai4fly-venv
    ~/ai4fly-venv/bin/pip install pymavlink pyserial
    sudo usermod -aG dialout $USER       # the serial port; log in again after
    git clone <this repository> ~/ai4fly-estimation

Raspberry Pi OS (Bookworm and later) refuses `pip install` into the system
Python, so pymavlink goes into a virtual environment that still sees the
system's numpy and pandas. Run the process with `~/ai4fly-venv/bin/python`,
as the service does; plain `python3` will not find pymavlink.

The trained weights are not in the repository. On a machine with torch, fit
the model's last-layer posterior -- the epistemic term needs it -- then
export, and copy `deploy/weights/quad.npz` (about 1.2 MB, most of it the
posterior) to the same path on the Pi:

    python quad_sim/laplace.py         # on the training machine
    python deploy/export.py
    scp deploy/weights/quad.npz <user>@<host>:ai4fly-estimation/deploy/weights/

Nothing on the Pi needs torch. If the posterior is missing from the export,
the process says so on startup and runs without the epistemic term.

## Wiring

CubeOrange TELEM2 to the Pi's primary UART:

    Cube TELEM2 TX  ->  Pi GPIO 15 (RXD, pin 10)
    Cube TELEM2 RX  <-  Pi GPIO 14 (TXD, pin 8)
    GND             --  GND

Both sides are 3.3 V logic. Enable the UART on the Pi (`raspi-config` →
Interface Options → Serial Port: no login shell, yes hardware serial), which
makes it available as `/dev/serial0`.

## ArduPilot parameters

On the Cube, for TELEM2 (SERIAL2):

    SERIAL2_PROTOCOL  2        MAVLink 2
    SERIAL2_BAUD      921      921600 baud

`pi_main.py` requests `SCALED_IMU` at 50 Hz itself through
`MAV_CMD_SET_MESSAGE_INTERVAL`, so the stream-rate parameters (`SR2_*`)
can stay at their defaults.

## Confirming the link before enabling the service

    ~/ai4fly-venv/bin/python deploy/pi_main.py --link /dev/serial0:921600 --once --max-steps 500

should print the attitude it started from, run 500 steps, and leave a log
in `deploy/logs/`. If it prints `warning: no flight controller` instead, the
message says what it did hear: nothing at all points at the wiring or the
baud rate; an ArduPilot flight controller with another system id means
`--system` should be its `SYSID_THISMAV` (default 1). Only that flight
controller's messages are used; anything else ArduPilot forwards onto the
link — a ground station, a gimbal, another vehicle — is ignored.
Open the log and check that `dt` sits near 0.02 and that `updated` is 1 on
nearly every row; a link delivering at 2 Hz rather than 50 shows as
`updated` being mostly 0, and means the rate request was not honoured.

In the ground station, the named values `hb_accel`, `hb_gyro`, `hb_mag`,
`hn_accel`, `hn_gyro`, `hn_mag`, `nis`, `step_ms`, `nov_accel`, `nov_gyro`,
`nov_mag` and `integrity` should appear in the Quick or Status tab within a
few seconds, and in the dataflash log afterwards under `NVF`.

## When every sensor fails

`deploy/integrity.py` decides, every step, whether the estimate can be used
at all, and the process reports it as the named value `integrity`:

| value | state | meaning | alert |
|---|---|---|---|
| 0 | ok | | `sensors recovered` on the way back |
| 1 | degraded | a device has failed for a second | `sensor failed: <device> <why>` (warning) |
| 2 | invalid | every device has failed for a second, the readings have stopped for two, or the estimate has gone non-finite | `ESTIMATE INVALID: <why>` (critical) |

A device has *failed* only when its readings carry no information: frozen
for half a second, not finite, impossible, or a magnetometer reporting no
field. None of that depends on the learned model, on purpose — on the first
flights the model-based signals are expected to be high everywhere, because
the real vehicle is not the simulated one, and that gap must not be what
sends the vehicle home. The model's signals mark a device *suspect* in the
log's `suspect` column and change nothing.

In simulation (`deploy/check_integrity.py`, `results/integrity.csv`) it
declares the estimate invalid within 1.1–2.0 s of every sensor freezing,
going dead, returning garbage or going silent, and never on healthy or
aggressive flights, one or two failed devices, or every device ten times
noisier.

On the bench, expect the odd `sensor failed: ... frozen` while the vehicle
sits perfectly still: `SCALED_IMU` carries whole milli-g and milli-rad/s, and
a quiet sensor at rest can repeat the same integers for half a second. In
flight, vibration makes every reading move.

### Asking for RTL

With `--allow-rtl` (quadcopter only), the process asks the flight controller
to switch to RTL when the estimate becomes invalid. It asks, and stands down
at the first sign it should:

- only of an ArduPilot multicopter or helicopter, the one with MAVLink
  system id `--system` (default 1, ArduPilot's `SYSID_THISMAV`) —
  everything else ArduPilot forwards onto this link, a ground station's or
  another vehicle's traffic included, is ignored;
- only while it is armed and in the air, by `EXTENDED_SYS_STATE`'s landed
  state (requested at 2 Hz), with a heartbeat in the last three seconds;
- never while it is already in RTL, SmartRTL, Auto RTL or Land;
- at most three requests, two seconds apart, until one is acknowledged;
- a refusal is announced and not retried;
- once per flight: if RTL engages and the pilot switches out of it, it does
  not ask again until the vehicle has disarmed, even if the link drops and
  reconnects in between;
- whenever one of these holds a request back while the estimate is
  invalid, it prints which, once per flight.

The request goes back over `--link` itself. Leave `--sink` out when RTL
requests are on: a separate sink cannot hear the acknowledgement.

Two things limit what this can do, and both are deliberate. The Pi sees one
of the CubeOrange's three IMUs through `SCALED_IMU`; ArduPilot flies on all
three and switches away from a bad one, so "every sensor on the Pi failed"
can mean one IMU or the link failed while the vehicle is fine — RTL is then
a conservative choice, not a necessary one. And RTL needs a position
estimate: if the failure is real enough to take that away, ArduPilot will
refuse the request, and its own EKF failsafe (`FS_EKF_ACTION`), which lands,
is what protects the vehicle. This request is a second opinion, never the
primary failsafe.

**Try it in SITL before it goes near a vehicle.** With ArduPilot's
simulator running and streaming to this machine:

    sim_vehicle.py -v ArduCopter --out udp:127.0.0.1:14550
    python3 deploy/pi_main.py --link udpin:0.0.0.0:14550 --allow-rtl

arm and take off in Loiter, then make the sensors the Pi reads fail in the
simulator: SITL has parameters that fail or freeze its simulated IMUs and
compass — list them in MAVProxy with `param show SIM_*FAIL*` and
`param show SIM_*STUCK*`, since the names have changed between ArduPilot
versions — and fail the first IMU and the compass. Within about two seconds
of every reading freezing, the process should print `integrity: invalid
(every sensor failed)` and `return home: RTL requested`, then `RTL request
accepted`, and the simulated vehicle should switch to RTL. Switch it back to
Loiter and check that it prints `RTL left` and asks nothing more. Then land,
disarm, and check that a second flight would be asked for again.

Stopping the `SCALED_IMU` stream instead is not a test of this: the process
asks for the stream again within a second, which is what keeps it alive
across a flight-controller reboot. None of this has been run yet.

## Running at boot

    sed "s#/home/pi#$HOME#g; s#^User=pi#User=$USER#" deploy/pi/estimator.service \
        | sudo tee /etc/systemd/system/estimator.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now estimator
    journalctl -u estimator -f

The `sed` fills in your user and home directory: the unit is written for a
user called `pi`, which Raspberry Pi OS no longer creates.

A log runs to about 140 MB an hour, since every row carries the raw reading,
the estimate and the three noise terms. Each time the service starts it
compresses the logs earlier runs left in `deploy/logs/` to `.csv.gz`, about
a third the size; `pandas.read_csv` reads them unchanged. Nothing is ever
deleted automatically -- the first flights are the data this whole project
is waiting for -- so copy logs off and clear the folder now and then.

## What to look at in the first logs

`step_ms` is the per-step cost on this hardware, which every compute claim in
the main README currently lacks. The healthy-flight NIS should settle near 9
after the first second; if it runs high throughout, the sensor units or
frame differ from what the model was trained on, and
`deploy/mavlink_source.py` is where the conversion lives. A `repaired`
column that is ever nonzero means the covariance lost definiteness and was
floored or reset; on simulated flights it never is.

The health values are reported but not alarmed on the quadcopter. In
simulation they do not read as severities (`quad_sim/health_readout.py`):
entries above 2 appear on healthy flights and a severity-3 accelerometer
fault moves its entry to under 1. Log them, do not act on them, and expect
the first flights to add a sim-to-hardware gap on top of that.

The novelty values are the one channel built to measure that gap. Their
alert levels come from healthy simulated flights (`deploy/calibrate_novelty.py`,
written to `results/novelty_threshold.csv`), so on the first real flights
they answer one question: does the real vehicle look like the simulated one
to the model? A novelty alert on a healthy first flight is not a fault. It is
the sim-to-real gap, measured per device, and it says which device the
model was trained on least faithfully. Fly the first flights gently, inside
the envelope the model was trained on, and recalibrate the levels from those
logs before treating an alert as a warning.
