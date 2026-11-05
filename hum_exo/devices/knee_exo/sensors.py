"""Hardware input schema, independent of the simulator and training stack."""

IMU_NAMES = ("left_thigh", "right_thigh", "left_shank", "right_shank")
INPUT_NAMES = (
    *(f"{part}.{kind}_{axis}" for part in IMU_NAMES for kind in ("acc", "gyro") for axis in ("x", "y", "z")),
    "right_knee.angle",
    "right_knee.angular_velocity",
)
INPUT_UNITS = (*(unit for _ in IMU_NAMES for unit in (*("m/s^2",) * 3, *("rad/s",) * 3)), "rad", "rad/s")
