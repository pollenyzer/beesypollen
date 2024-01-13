"""Module contains custom error definitions that contain error codes that could
be passed to the front end.
"""


class BackgroundError(Exception):
    code = "E01"
    msg = "No white background detected. Please place the pollen on a white background."

    def __init__(self, *args: object) -> None:
        super().__init__(self.msg)


class NoPollenError(Exception):
    code = "E02"
    msg = "No pollen found. Please ensure that your input contains pollen and is suited for detection."

    def __init__(self, *args: object) -> None:
        super().__init__(self.msg)


class CalibrationError(Exception):
    code = "E03"
    msg = "Calibration failed. Please ensure that you are using the correct color checker."

    def __init__(self, *args: object) -> None:
        super().__init__(self.msg)
