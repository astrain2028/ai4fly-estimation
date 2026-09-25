# Running the estimator on a Raspberry Pi 5 with a CubeOrange

Status: the software has been run against a simulated vehicle over a UDP
loopback and against replayed simulation data. It has not been run against a
CubeOrange. Everything below about wiring and ArduPilot parameters is from
the ArduPilot documentation for companion computers and has not been
confirmed on this hardware. Confirm each step before enabling the service.

## What runs

One Python process, `deploy/pi_main.py`. It reads `SCALED_IMU` from the
flight controller over MAVLink, steps the estimator in `deploy/runtime.py`,
sends six health estimates and the filter's NIS back as `NAMED_VALUE_FLOAT`
once a second, raises a `STATUSTEXT` alert when a health estimate or NIS
crosses a threshold, and writes one log row per step to `deploy/logs/`.

The estimator is advisory. Nothing it sends feeds the flight controller's
EKF; the vehicle flies on its own estimate whether this process is running
or not. That is the configuration to fly first.

Measured on a laptop: about 1.7 ms per step for the quadcopter against a
20 ms budget at 50 Hz, and 28 MB resident with the weights exported to
numpy. The Pi 5 will be slower; measure it there with `deploy/runtime.py`
before trusting the margin.

## Software on the Pi

    sudo apt install python3-numpy python3-pandas
    pip3 install pymavlink
    git clone <this repository> ~/ai4fly-estimation

The trained weights are not in the repository. Export them on a machine
with torch and copy `deploy/weights/quad.npz` (85 KB) to the same path on
the Pi:

    python deploy/export.py            # on the training machine
    scp deploy/weights/quad.npz pi@<host>:ai4fly-estimation/deploy/weights/

Nothing on the Pi needs torch.

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

    python3 deploy/pi_main.py --link /dev/serial0:921600 --once --max-steps 500

should print a heartbeat, run 500 steps, and leave a log in `deploy/logs/`.
Open the log and check that `dt` sits near 0.02 and that `updated` is 1 on
nearly every row; a link delivering at 2 Hz rather than 50 shows as
`updated` being mostly 0, and means the rate request was not honoured.

In the ground station, the named values `hb_accel`, `hb_gyro`, `hb_mag`,
`hn_accel`, `hn_gyro`, `hn_mag`, `nis` and `step_ms` should appear in the
Quick or Status tab within a few seconds, and in the dataflash log
afterwards under `NVF`.

## Running at boot

    sudo cp deploy/pi/estimator.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now estimator
    journalctl -u estimator -f

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
