from os.path import join, dirname
import numpy as np
from skimage.feature import peak_local_max
import cv2
import tensorflow.keras as keras
from keras.models import load_model, Model
from typing import Union
from pathlib import Path
from .utils import (
    standardize_pollen_image,
    L,
)

# these values do (in principal) depend on the pollen detection model
STANDARDIZED_WIDTH_M11_02 = 1104 * 2
STANDARDIZED_HEIGHT_M11_02 = 784 * 2
STANDARDIZED_POLLEN_DIM_M11_02 = 10

STANDARDIZED_WIDTH_M11_06 = 1104 * 2
STANDARDIZED_HEIGHT_M11_06 = 784 * 2
STANDARDIZED_POLLEN_DIM_M11_06 = 10


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def detect_pollen_m11_06(
    img_raw_rgb: np.ndarray,
    log_dir: Union[str, None],
    preloaded_model: Model = None
) -> np.ndarray:
    """Applies the given pollen detection model on the given image and returns
    for a list of xy detections. This is the most up to date pollen detection
    model available.

    Args:
        img_raw_rgb (np.ndarray): The rgb(!) image to be analyzed.
        log_dir (Union[str, None]): If `log_dir` is not None, image logs are
            placed to the specified directory.

    Returns:
        np.ndarray: A tuple of (1) the coordinates of the upper left corner
        of the standardized image (2) the upper right corner of the
        standardized image, (3) a nx2 array that contains for n detections
        the x and y coordinates in relation to the input image and (4) the
        standardized image.
    """
    # load model
    model_path = join(
        dirname(__file__),
        "models",
        "m11_06",
        "2022-12-14T00-24-22",
        "checkpoints",
        "pollen_detection_m11_06_epoch11_vl0.14146"
    )

    if preloaded_model:
        model = preloaded_model
    else:
        model = load_model(model_path)
    mono = False

    (
        matrix_scale_down,
        matrix_scale_up,
        std_upper_left,
        std_lower_right,
        std_img,
    ) = standardize_pollen_image(
        img_raw_rgb=img_raw_rgb,
        mono=mono,
        standardized_height=(
            STANDARDIZED_HEIGHT_M11_06
            if img_raw_rgb.shape[0] < img_raw_rgb.shape[1]
            else STANDARDIZED_WIDTH_M11_06
        ),
        standardized_width=(
            STANDARDIZED_HEIGHT_M11_06
            if img_raw_rgb.shape[0] >= img_raw_rgb.shape[1]
            else STANDARDIZED_WIDTH_M11_06
        ),
    )
    img_raw_rgb.shape

    pollen_detection_heatmap = sigmoid(model.predict(std_img[None, ...]))

    if log_dir is not None:
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        # cv2.imwrite(join(log_dir, f"0_raw.jpg"), img_raw_rgb[..., ::-1])
        cv2.imwrite(join(log_dir, f"1_std.jpg"), std_img[..., ::-1])
        cv2.imwrite(
            join(log_dir, f"2_pred.jpg"),
            pollen_detection_heatmap[0, :, :, 0] * 255,
        )

    local_maxima = peak_local_max(
        pollen_detection_heatmap[0, :, :, 0],
        threshold_abs=0.2,
        min_distance=10,
        exclude_border=STANDARDIZED_POLLEN_DIM_M11_06,
    )

    if len(local_maxima) == 0:
        L.info("No pollen found.")
        return (
            std_upper_left,
            std_lower_right,
            np.empty((0, 2), np.int32),
            std_img
        )

    # find pollen centers on a unet detection heatmap
    pollen_detections_xy = cv2.transform(
        local_maxima.reshape((-1, 2)).astype(np.int32)[None, :, ::-1],
        matrix_scale_up,
    )[0, :, 0:2]

    if log_dir is not None:
        annotated_img = img_raw_rgb.copy()
        for x, y in pollen_detections_xy:
            cv2.circle(
                img=annotated_img,
                center=(x, y),
                radius=int(annotated_img.shape[0] / 250),
                thickness=-1,
                color=(100, 100, 255),
            )
        cv2.imwrite(
            join(log_dir, f"3_annotated.jpg"),
            cv2.cvtColor(
                cv2.addWeighted(img_raw_rgb, 0.5, annotated_img, 1 - 0.5, 0),
                cv2.COLOR_RGB2BGR
            )
        )

    L.info(f"{len(pollen_detections_xy)} pollen detected.")

    return (std_upper_left, std_lower_right, pollen_detections_xy, std_img)


def detect_pollen_m11_02(
    img_raw_rgb: np.ndarray,
    log_dir: Union[str, None],
    preloaded_model: Model = None
) -> np.ndarray:
    """Applies the given pollen detection model on the given image and returns
    for a list of xy detections.

    Args:
        img_raw_rgb (np.ndarray): The rgb(!) image to be analyzed.
        log_dir (Union[str, None]): If `log_dir` is not None, image logs are
            placed to the specified directory.

    Returns:
        np.ndarray: A tuple of (1) the coordinates of the upper left corner
        of the standardized image (2) the upper right corner of the
        standardized image, (3) a nx2 array that contains for n detections
        the x and y coordinates in relation to the input image and (4) the
        standardized image.
    """
    # load model
    model_path = join(
        dirname(__file__),
        "models",
        "m11_02/2022-12-02T17-44-36",
        "checkpoints",
        "pollen_detection_m11_02_epoch04_vl0.13650"
    )

    if preloaded_model:
        model = preloaded_model
    else:
        model = load_model(model_path)
    mono = False

    (
        matrix_scale_down,
        matrix_scale_up,
        std_upper_left,
        std_lower_right,
        std_img,
    ) = standardize_pollen_image(
        img_raw_rgb=img_raw_rgb,
        mono=mono,
        standardized_height=STANDARDIZED_HEIGHT_M11_02,
        standardized_width=STANDARDIZED_WIDTH_M11_02
    )

    pollen_detection_heatmap = sigmoid(model.predict(std_img[None, ...]))

    if log_dir is not None:
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        # cv2.imwrite(join(log_dir, f"0_raw.jpg"), std_img)
        cv2.imwrite(join(log_dir, f"1_std.jpg"), std_img)
        cv2.imwrite(
            join(log_dir, f"2_pred.jpg"),
            pollen_detection_heatmap[0, :, :, 0] * 255,
        )

    local_maxima = peak_local_max(
        pollen_detection_heatmap[0, :, :, 0],
        threshold_abs=0.2,
        min_distance=10,
        exclude_border=STANDARDIZED_POLLEN_DIM_M11_02,
    )

    if len(local_maxima) == 0:
        L.info("No pollen found.")
        return (
            std_upper_left,
            std_lower_right,
            np.empty((0, 2), np.int32),
            std_img
        )

    # find pollen centers on a unet detection heatmap
    pollen_detections_xy = cv2.transform(
        local_maxima.reshape((-1, 2)).astype(np.int32)[None, :, ::-1],
        matrix_scale_up,
    )[0, :, 0:2]

    if log_dir is not None:
        annotated_img = img_raw_rgb.copy()
        for x, y in pollen_detections_xy:
            cv2.circle(
                img=annotated_img,
                center=(x, y),
                radius=int(annotated_img.shape[0] / 250),
                thickness=-1,
                color=(100, 100, 255),
            )
        cv2.imwrite(
            join(log_dir, f"3_annotated.logdir"),
            cv2.cvtColor(
                cv2.addWeighted(img_raw_rgb, 0.5, annotated_img, 1 - 0.5, 0),
                cv2.COLOR_RGB2BGR
            )
        )

    L.info(f"{len(pollen_detections_xy)} pollen detected.")

    return (std_upper_left, std_lower_right, pollen_detections_xy)
