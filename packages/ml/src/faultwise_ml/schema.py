"""Canonical positional schema shared by C-MAPSS variants FD001–FD004."""

ID_COLUMNS = ("unit", "cycle")
SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
INPUT_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
