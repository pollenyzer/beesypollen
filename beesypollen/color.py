import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import cv2
import pandas as pd
from typing import Dict
from .utils import standardize_pollen_image, L


class COLOR_EXTRACTION_CRIT:
    SIZE = 0
    LIGHTNESS = 1


class COLOR_EXTRACTION_METHOD:
    KMEANS = 0
    GMM = 1


def _extract_single_pollen_color_rgb(
    x: int,
    y: int,
    img_rgb: np.ndarray,
    pollen_dim: int,
    primary_crit: COLOR_EXTRACTION_CRIT = COLOR_EXTRACTION_CRIT.LIGHTNESS,
    method: COLOR_EXTRACTION_METHOD = COLOR_EXTRACTION_METHOD.GMM
) -> Dict:
    """Returns a primary and secondary pollen color.

    This function assumes that a pollen consists of exactly two colors.
    A primary, brighter color and a secondary, darker color. This is
    argued with the observation of shadows on spherical pollen and fits
    well with real data. A Gaussian Mixture Model (or kmeans) is fitted
    to find the rgb values for primary and secondary colors.

    Args:
        x (int): x coordinate of pollen.
        y (int): y coordinate of pollen.
        img_rgb (np.ndarray): The (rgb) source image.
        pollen_dim (int): The pollen dimension in px.
        primary_crit (COLOR_EXTRACTION_CRIT): If "size", the primary color is
            the biggest color cluster, if "lightness", the primary color is
            the lighter color. The latter is usefull if primary and secondary
            colors should distinguish shadow from non-shadow, the former is
            the better choice for all other cases. Defaults to "lightness".
        method (COLOR_EXTRACTION_METHOD): If "KMEANS", the method for
            clustering pixels is kmeans. If "GMM", a gaussian mixture model
            is applied. KMeans is approx. 3 times faster.

    Returns:
        dict: A dictionary that contains primary and secondary colors in
        rgb color space.
    """
    polle_cropped_rgb = img_rgb[
        y - pollen_dim // 2:y - pollen_dim // 2 + pollen_dim,
        x - pollen_dim // 2:x - pollen_dim // 2 + pollen_dim,
    ]

    polle_cropped_lab = cv2.cvtColor(polle_cropped_rgb, cv2.COLOR_RGB2LAB)

    px_list_lab = polle_cropped_lab.reshape([-1, 3])

    if method == COLOR_EXTRACTION_METHOD.GMM:
        GMM = GaussianMixture(
            n_components=2,
            covariance_type="full",
        )
        predicted_labels = GMM.fit_predict(px_list_lab)
        extracted_colors_lab = GMM.means_.astype(np.uint8)
    elif method == COLOR_EXTRACTION_METHOD.KMEANS:
        KM = KMeans(2, max_iter=10)
        predicted_labels = KM.fit_predict(px_list_lab)
        extracted_colors_lab = KM.cluster_centers_.astype(np.uint8)
    else:
        raise NotImplemented(f"Method {method} not implemented.")

    extracted_colors_rgb = cv2.cvtColor(
        extracted_colors_lab[None], cv2.COLOR_LAB2RGB
    )[0]
    assert extracted_colors_rgb.dtype == np.uint8

    if primary_crit == COLOR_EXTRACTION_CRIT.LIGHTNESS:
        if extracted_colors_lab[0, 0] > extracted_colors_lab[1, 0]:
            # order by lightness
            prim_rgb = extracted_colors_rgb[0]
            scnd_rgb = extracted_colors_rgb[1]
        else:
            prim_rgb = extracted_colors_rgb[1]
            scnd_rgb = extracted_colors_rgb[0]
    elif primary_crit == COLOR_EXTRACTION_CRIT.SIZE:
        if (predicted_labels == 0).sum() > (predicted_labels == 1).sum():
            # order by label frequency
            prim_rgb = extracted_colors_rgb[0]
            scnd_rgb = extracted_colors_rgb[1]
        else:
            prim_rgb = extracted_colors_rgb[1]
            scnd_rgb = extracted_colors_rgb[0]
    else:
        raise NotImplementedError()

    result = dict(
        [(f"primary_{c}", int(prim_rgb[cid])) for cid, c in enumerate("rgb")]
        + [
            (f"secondary_{c}", int(scnd_rgb[cid]))
            for cid, c in enumerate("rgb")
        ]
    )
    return result


def extract_pollen_color(
    img_raw_rgb: np.ndarray,
    pollen_detections_xy: np.ndarray,
    pollen_dim: int,
    standardized_height: int,
    standardized_width: int,
    primary_crit: COLOR_EXTRACTION_CRIT = COLOR_EXTRACTION_CRIT.LIGHTNESS,
    method: COLOR_EXTRACTION_METHOD = COLOR_EXTRACTION_METHOD.GMM,
) -> np.ndarray:
    """Extracts the pollen color for each pollen detection.

    Args:
        img_raw_rgb (np.ndarray): The rgb(!) image to be analyzed.
        pollen_detections_xy (np.ndarray): The nx2 array of pollen detections.
        pollen_dim (int): Denotes the area of pixels that are considered when
            extracting the pollen color.
        standardized_height (int): The height for standardization.
        standardized_width (int): The width for standardization.
        primary_crit (COLOR_EXTRACTION_CRIT): If "size", the primary color is
            the biggest color cluster, if "lightness", the primary color is
            the lighter color. The latter is usefull if primary and secondary
            colors should distinguish shadow from non-shadow, the former is
            the better choice for all other cases. Defaults to "lightness".
        method (COLOR_EXTRACTION_METHOD): If "KMEANS", the method for
            clustering pixels is kmeans. If "GMM", a gaussian mixture model
            is applied. KMeans is approx. 3 times faster.
    Returns:
        pd.DataFrame: A pd.Dataframe object that contains for each pollen the
        primary and secondary pollen color.
    """
    if len(pollen_detections_xy) == 0:
        # return empty pollen color dataframe
        columns = [f"primary_{c}" for c in "rgb"] + [
            f"secondary_{c}" for c in "rgb"
        ]
        return pd.DataFrame(columns=columns, dtype=np.uint8)

    (
        matrix_scale_down,
        matrix_scale_up,
        std_upper_left,
        std_lower_right,
        std_img,
    ) = standardize_pollen_image(
        img_raw_rgb=img_raw_rgb,
        mono=False,
        standardized_height=standardized_height,
        standardized_width=standardized_width
    )
    assert (std_img >= 0).all()

    standardized_detections_xy = cv2.transform(
        pollen_detections_xy[None],
        matrix_scale_down,
    )[0, :, 0:2]

    pollen_colors = [
        _extract_single_pollen_color_rgb(
            x=x,
            y=y,
            img_rgb=std_img,
            pollen_dim=pollen_dim,
            primary_crit=primary_crit,
            method=method
        )
        for x, y in standardized_detections_xy
    ]

    df_pollen_colors = pd.DataFrame.from_dict(pollen_colors, dtype=np.uint8)
    L.info(f"{len(df_pollen_colors)} pollen colors extracted.")

    return df_pollen_colors
