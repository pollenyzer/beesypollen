import numpy as np
import cv2
import logging
import pandas as pd
from os.path import join
from pathlib import Path
from typing import Union
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

L = logging.getLogger("pollenyzer")
logging.basicConfig(level=logging.INFO)

DINA5_LANDSCAPE_WIDTH_MM = 210
DINA5_LANDSCAPE_HEIGHT_MM = 148


def viz_dendrogram(
    primary_rgb_colors: np.ndarray,
    linkage_matrix: np.ndarray,
    log_dir: Union[str, None],
):
    """A helper function that creates a dendrogram and returns the matplotlib
    figure.

    Args:
        primary_rgb_colors (np.ndarray): The primary pollen colors (rbg).
        linkage_matrix (np.ndarray): The pre calculated linkage matrix.
        log_dir (Union[str, None]): The optional logging directory.

    Returns:
        mpl.Figure: The dendrogram figure object.
    """
    if len(primary_rgb_colors) == 0:
        # nothing to visualize
        return None

    N = len(primary_rgb_colors)
    label_colors = np.c_[primary_rgb_colors / 255, np.ones(N)]

    # A dendrogram plot is created for display in the frontend.
    fig = plt.figure(figsize=(5, 8 / 35 * N))
    ax = plt.gca()
    dendrogram(
        linkage_matrix,
        color_threshold=20 * 2,
        above_threshold_color="#EEE",
        leaf_font_size=46,
        ax=ax,
        orientation="right",
    )

    # set label color to (primary) pollen color
    lbl_ids = []
    for lbl in ax.get_ymajorticklabels():
        lbl_id = int(lbl.get_text())
        lbl_ids.append(lbl_id)
        lbl.set_color(label_colors[lbl_id])
    ax.set_yticklabels(["⚫"] * N)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    plt.tight_layout()
    if log_dir is not None:
        fig.savefig(join(log_dir, "dendrogram.pdf"))
    return fig


def pollen_export_csv(
    df_pollen_colors: pd.DataFrame,
    pollen_detections_xy: np.ndarray,
    cluster_division: np.ndarray,
    log_dir: str,
):
    """A helper function that exports the analysis results into a csv file.

    Args:
        df_pollen_colors (pd.DataFrame): The extracted pollen colors.
        pollen_detections_xy (np.ndarray): The pollen detections.
        cluster_division (np.ndarray): The best cluster division found.
        log_dir (str): The place where to store the csv
    """
    df = df_pollen_colors.copy()
    df["pos_x"] = pollen_detections_xy[:, 0]
    df["pos_y"] = pollen_detections_xy[:, 1]
    df["proposed_cluster_id"] = cluster_division
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    out_path = join(log_dir, "result.csv")
    L.info(f"CSV file written to {out_path}.")
    df.to_csv(out_path, ";")


def coord_inside_img(
    x: int, y: int, upper_left_xy: np.ndarray, lower_right_xy: np.ndarray
) -> bool:
    """Helper function that return True if a coordinate is inside the
    rectangle definded by upper left and lower right coordinate.

    Args:
        x (int): x coordinate of pollen.
        y (int): y coordinate of pollen.
        upper_left_xy (np.ndarray): The upper left corner to check on.
        lower_right_xy (np.ndarray): The lower right corner to check on.

    Returns:
        bool: True, if coordinate is inside the upper_left and lower right
            coordinates.
    """
    return not (
        x <= upper_left_xy[0]
        or x >= lower_right_xy[0]
        or y <= upper_left_xy[1]
        or y >= lower_right_xy[1]
    )


def standardize_pollen_image(
    img_raw_rgb: np.ndarray,
    mono: bool,
    standardized_height,
    standardized_width
) -> np.ndarray:
    """Fits a centered DINA5 paper into the input image `img_raw_rgb`.
    This area is then scaled to a STANDARDIZED_WIDTH and
    STANDARDIZED_HEIGHT.

    Args:
        img_raw_rgb (np.ndarray): The image (rgb) to be standardized.
        mono (bool): If True, a mono channel image is returned.
        standardized_height (int): The height to scale to.
        standardized_width (int): The width to scale to.

    Returns:
        Tuple: A Tuple of (1) the matrix for scaling down, (2) the matrix
            for scaling up, (3) the upper left corner of the standardized
            image, (4) the lower right corner of the standardized im age,
            (5) the standardized image.
    """
    # A mono channel version of the image that is fed to the detector.
    # Averaging red and blue channels helps to easily distinct pollen
    # from white background.
    if mono:
        img = img_raw_rgb[:, :, 0:3].mean(axis=2)
    else:
        img = img_raw_rgb

    # Since the user is expected to photograph pollen on a DINA5 sheet, the
    # image is cropped to fit best to the expected DINA5 ratio.
    input_height, input_width = img.shape[0], img.shape[1]

    landscape = input_height < input_width

    dina5_height, dina5_width = (
        # landscape versus portrait
        (DINA5_LANDSCAPE_HEIGHT_MM, DINA5_LANDSCAPE_WIDTH_MM)
        if landscape
        else (DINA5_LANDSCAPE_WIDTH_MM, DINA5_LANDSCAPE_HEIGHT_MM)
    )

    # The raw image gets a DINA5 ratio by cropping either from left and right
    # or from top and bottom.
    dina5ratio = dina5_width / dina5_height
    if input_width / input_height < dina5ratio:
        # too high
        cropped_width = input_width
        cropped_height = int(cropped_width / dina5ratio)
    else:
        # too wide
        cropped_height = input_height
        cropped_width = int(cropped_height * dina5ratio)

    target_height, target_width = (
        (
            min(standardized_height, standardized_width),
            max(standardized_height, standardized_width)
        )
        if landscape
        else (
            max(standardized_height, standardized_width),
            min(standardized_height, standardized_width)
        )
    )

    # The detection model expects the input of a specific size. The scaling
    # is performed as a perspective transformation where each coordinate of
    # pts1 is transformed to the corresponding coordinate in pts2.
    pts1 = np.float32([
        (
            input_width / 2 - cropped_width / 2,
            input_height / 2 - cropped_height / 2
        ),
        (
            input_width / 2 + cropped_width / 2,
            input_height / 2 - cropped_height / 2
        ),
        (
            input_width / 2 + cropped_width / 2,
            input_height / 2 + cropped_height / 2
        ),
        (
            input_width / 2 - cropped_width / 2,
            input_height / 2 + cropped_height / 2
        ),
    ])

    std_upper_left, std_lower_right = pts1[0], pts1[2]

    pts2 = np.float32(
        [
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height],
        ]
    )

    matrix_scale_down = cv2.getPerspectiveTransform(pts1, pts2)
    matrix_scale_up = cv2.getPerspectiveTransform(pts2, pts1)

    # std_img has the shape and channels as required by the detection model.
    std_img = cv2.warpPerspective(
        src=img, M=matrix_scale_down, dsize=(target_width, target_height)
    )
    L.info(
        f"Image of shape {img_raw_rgb.shape} standardized to image of shape "
        f"{std_img.shape}."
    )
    return (
        matrix_scale_down,
        matrix_scale_up,
        std_upper_left,
        std_lower_right,
        std_img,
    )
