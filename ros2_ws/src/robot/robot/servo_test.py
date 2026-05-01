from __future__ import annotations
import time

from robot.robot import FirmwareState, Robot, Unit

TAG_ID = 13
POSITION_UNIT = Unit.MM

def configure_robot(robot: Robot) -> None:
    robot.set_unit(POSITION_UNIT)
    robot.set_tracked_tag_id(TAG_ID)

def run(robot: Robot) -> None:
    configure_robot(robot)
    robot.set_state(FirmwareState.RUNNING)

    print("Enabling servo on channel 1...")
    robot.enable_servo(1)
    time.sleep(0.5)

    print("Sweeping 0 -> 180 degrees")
    for angle in range(0, 181, 10):
        print(f"  Angle: {angle}")
        robot.set_servo(1, angle)
        time.sleep(0.3)

    print("Sweeping 180 -> 0 degrees")
    for angle in range(180, -1, -10):
        print(f"  Angle: {angle}")
        robot.set_servo(1, angle)
        time.sleep(0.3)

    print("Centering at 90 degrees")
    robot.set_servo(1, 90)
    time.sleep(0.5)

    robot.disable_servo(1)
    print("Done.")
    robot.shutdown()