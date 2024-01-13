import numpy as np
import pandas as pd
import pollenyzer.errors as errors


class Pollenyzer:
    def __init__(self) -> None:
        pass

    def dummy_pollen_result(self, input_img: np.ndarray):
        """return dummy results for testing"""
        h, w = input_img.shape[0:2]
        pollen_count = np.random.randint(10, 20)
        df = pd.DataFrame(index=np.arange(pollen_count))
        df[["y", "x"]] = np.random.randint(
            [0, 0], input_img.shape[0:2], size=(pollen_count, 2)
        )
        df[["r", "g", "b"]] = np.random.randint(0, 256, (pollen_count, 3))
        return df

    def dummy_error(self) -> Exception:
        return getattr(
            errors,
            np.random.choice(list(filter(lambda x: "Error" in x, dir(errors)))),
        )

    def dummy_color_checker_colors(self):
        return np.random.randint(0, 256, (24, 3)).tolist()

    def __call__(self, input_img: np.ndarray):
        """Performs automatic chromatic pollen analysis on images of pollen samples.

        Args:
            input_img (np.ndarray): Image containing corbicular pollen loads.

        Raises:
            BackgroundError: When no white background is detected.
            NoPollenError: If no pollen were detected.
            CalibrationError: If calibration failed.

        Returns:
            dict: A dictionary holding the serializable results.
        """
        # check input image type
        if not isinstance(input_img, np.ndarray):
            raise TypeError("Expected np.ndarray, but type {} found.").format(
                type(input_img)
            )

        # check input image dimension
        if input_img.ndim != 3:
            raise ValueError(
                "Expected 3 dimensions, but {} dimensions found.".format(input_img.ndim)
            )

        # check input image channels
        if input_img.shape[-1] not in (3, 4):
            raise ValueError(
                "Expected 3 channel (RGB) or 4 channel (RGBA), but {} channels found.".format(
                    len(input_img.shape)
                )
            )

        # handle 4-channel png files by ignoring alpha
        if input_img.shape[-1] == 4:
            input_img = input_img[:, :, 0:3]

        # fake data for testing
        pollen_detections = self.dummy_pollen_result(input_img)
        detected_color_checker_colors = self.dummy_color_checker_colors()
        calibrated_color_checker_colors = self.dummy_color_checker_colors()

        res = {
            "pollen_detections": pollen_detections.to_dict(orient="index"),
            "color_checker": {
                "detected_colors": detected_color_checker_colors,
                "calibrated_colors": calibrated_color_checker_colors,
            },
            "img_height": input_img.shape[0],
            "img_width": input_img.shape[1],
        }

        # flip a coin and raise error for testing
        if np.random.rand() > 0.5:
            raise self.dummy_error()

        return res
