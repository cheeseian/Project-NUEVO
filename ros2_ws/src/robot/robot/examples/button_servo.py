"""
button_servo.py — servo pulse + DC motor lift + jogged servo
=============================================================
BTN_1 pulses servo CH_16: A → B (rest SETTLE_S) → A.
BTN_7 / BTN_10 drive DC Motor 3 up/down while held.
BTN_6 / BTN_8  jog servo CH_13 while held (software angle limits).
BTN_2 disables all actuators and exits cleanly.

HOW TO RUN
----------
Copy this file over main.py, then restart the robot node:

    cp examples/button_servo.py main.py
    ros2 run robot robot

WHAT THE ROBOT DOES
-------------------
  BTN_1:  pulse CH_16  A → B → (rest) → A
  BTN_7:  hold to drive M3 up   (+200 PWM)
  BTN_10: hold to drive M3 down (-200 PWM)
  BTN_6:  hold to jog  CH_13 toward SERVO2_MAX_DEG
  BTN_8:  hold to jog  CH_13 toward SERVO2_MIN_DEG
  BTN_2:  disable all actuators and shut down

  BLUE LED  — idle at servo CH_16 position A
  GREEN LED — servo CH_16 at position B during the rest

CONFIGURATION
-------------
Edit the constants below to match your build.
"""

from __future__ import annotations

import time

from robot.hardware_map import (
    Button,
    DCMotorMode,
    DEFAULT_FSM_HZ,
    LED,
    Motor,
    POSITION_UNIT,
    ServoChannel,
)
from robot.robot import FirmwareState, Robot


# ---------------------------------------------------------------------------
# Configuration — edit to match your build
# ---------------------------------------------------------------------------

SERVO = ServoChannel.CH_16   # servo channel
SERVO_A_DEG = 95.0           # home / resting position (degrees)
SERVO_B_DEG = 170.0          # target position (degrees)
SETTLE_S = 0.5               # hold time at B before returning to A

LIFT_MOTOR = Motor.DC_M3     # DC motor used as lift
LIFT_UP_PWM = 200            # PWM for upward motion
LIFT_DOWN_PWM = -200         # PWM for downward motion

SERVO2 = ServoChannel.CH_13  # jogged servo channel
SERVO2_MIN_DEG = 0.0         # software minimum angle
SERVO2_MAX_DEG = 180.0       # software maximum angle
SERVO2_INIT_DEG = 90.0       # starting angle
SERVO2_SPEED_DEG_S = 60.0    # jog speed in degrees per second

LED_BRIGHTNESS = 220


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configure_robot(robot: Robot) -> None:
    robot.set_unit(POSITION_UNIT)


def start_robot(robot: Robot) -> None:
    current = robot.get_state()
    if current in (FirmwareState.ESTOP, FirmwareState.ERROR):
        robot.reset_estop()
    robot.set_state(FirmwareState.RUNNING)


def all_leds_off(robot: Robot) -> None:
    for led in (LED.RED, LED.GREEN, LED.BLUE, LED.ORANGE, LED.PURPLE):
        robot.set_led(led, 0)


# ---------------------------------------------------------------------------
# run() — entry point called by the robot node
# ---------------------------------------------------------------------------

def run(robot: Robot) -> None:
    configure_robot(robot)
    start_robot(robot)
    all_leds_off(robot)

    robot.enable_servo(SERVO)
    robot.set_servo(SERVO, SERVO_A_DEG)
    robot.set_led(LED.BLUE, LED_BRIGHTNESS)

    robot.enable_motor(LIFT_MOTOR, DCMotorMode.PWM)
    robot.set_motor_pwm(LIFT_MOTOR, 0)

    robot.enable_servo(SERVO2)
    robot.set_servo(SERVO2, SERVO2_INIT_DEG)

    print("=" * 60)
    print("BUTTON SERVO + LIFT + JOG DEMO")
    print(f"  Servo CH_{SERVO}   |  A={SERVO_A_DEG:.0f}°  B={SERVO_B_DEG:.0f}°  rest={SETTLE_S}s  — BTN_1")
    print(f"  Lift  M{LIFT_MOTOR}    |  up=BTN_7 (+{LIFT_UP_PWM})  down=BTN_10 ({LIFT_DOWN_PWM})")
    print(f"  Servo CH_{SERVO2}  |  {SERVO2_MIN_DEG:.0f}°–{SERVO2_MAX_DEG:.0f}°  {SERVO2_SPEED_DEG_S:.0f}°/s  — BTN_6/BTN_8")
    print("  BTN_2: exit")
    print("=" * 60)

    last_lift_pwm = 0
    servo2_angle = SERVO2_INIT_DEG
    period = 1.0 / float(DEFAULT_FSM_HZ)
    step_per_tick = SERVO2_SPEED_DEG_S / DEFAULT_FSM_HZ
    next_tick = time.monotonic()

    while True:
        # ── Servo pulse ──────────────────────────────────────────────────────
        if robot.was_button_pressed(Button.BTN_1):
            robot.set_servo(SERVO, SERVO_B_DEG)
            robot.set_led(LED.BLUE, 0)
            robot.set_led(LED.GREEN, LED_BRIGHTNESS)
            print(f"[SERVO] → B ({SERVO_B_DEG:.0f}°)")

            time.sleep(SETTLE_S)

            robot.set_servo(SERVO, SERVO_A_DEG)
            robot.set_led(LED.GREEN, 0)
            robot.set_led(LED.BLUE, LED_BRIGHTNESS)
            print(f"[SERVO] → A ({SERVO_A_DEG:.0f}°)")

            robot.was_button_pressed(Button.BTN_1)  # consume any queued press
            next_tick = time.monotonic()

        # ── Lift hold-to-run ─────────────────────────────────────────────────
        up   = robot.get_button(Button.BTN_7)
        down = robot.get_button(Button.BTN_10)

        if up and not down:
            pwm = LIFT_UP_PWM
        elif down and not up:
            pwm = LIFT_DOWN_PWM
        else:
            pwm = 0

        if pwm != last_lift_pwm:
            robot.set_motor_pwm(LIFT_MOTOR, pwm)
            if pwm > 0:
                print(f"[LIFT] up   (+{pwm})")
            elif pwm < 0:
                print(f"[LIFT] down ({pwm})")
            else:
                print("[LIFT] stop")
            last_lift_pwm = pwm

        # ── Servo2 jog ───────────────────────────────────────────────────────
        jog_up   = robot.get_button(Button.BTN_6)
        jog_down = robot.get_button(Button.BTN_8)

        if jog_up and not jog_down:
            new_angle = min(servo2_angle + step_per_tick, SERVO2_MAX_DEG)
        elif jog_down and not jog_up:
            new_angle = max(servo2_angle - step_per_tick, SERVO2_MIN_DEG)
        else:
            new_angle = servo2_angle

        if new_angle != servo2_angle:
            servo2_angle = new_angle
            robot.set_servo(SERVO2, servo2_angle)

        # ── Exit ─────────────────────────────────────────────────────────────
        if robot.was_button_pressed(Button.BTN_2):
            robot.set_motor_pwm(LIFT_MOTOR, 0)
            robot.disable_motor(LIFT_MOTOR)
            robot.disable_servo(SERVO)
            robot.disable_servo(SERVO2)
            all_leds_off(robot)
            print("Shutting down.")
            robot.shutdown()
            return

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
