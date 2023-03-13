from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import cv2
import pandas as pd
import numpy as np
from .utils import L


LAB_HIERARCHICAL_COLOR_DIFFERENCE_THRESHOLD = 20


class ClusterMode():
    PRIMARY = 0
    PRIMARY_AND_SECONDARY = 1


def best_cluster(
    df_pollen_colors: pd.DataFrame,
    useCalibratedColors: bool,
    cluster_by: ClusterMode = ClusterMode.PRIMARY
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the best color clusters based on a LAB-color-difference threshold
        of 20 (educated guess) and a hierarchical cluster approach.

    Args:
        df_pollen_colors (pd.DataFrame): A dataframe that contains primary and
            secondary rgb colors (that is channels named primary_r for
            primary red channel and so forth).
        useCalibratedColors (bool): If true, the calibrated colors are used
            for clustering. Make sure that the calibrated colors exist in the
            dataframe that is passed to this function.
        cluster_by (ClusterMode): If ClusterMode.PRIMARY, then secondary
            colors are ignored. If ClusterMode.PRIMARY_AND_SECONDARY, then
            secondary colors influence the clustering as much as the primary
            colors. Defaults to ClusterMode.PRIMARY.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of (linkage_matrix,
            cluster_division).
    """
    if len(df_pollen_colors) == 0:
        # nothing to cluster
        return (None, np.empty((0), np.int32))
    elif len(df_pollen_colors) == 1:
        return (None, np.array([0], np.int32))

    calibrationPrefix = "calib_" if useCalibratedColors else ""

    X_primary_lab = cv2.cvtColor(
        df_pollen_colors[
            [f"primary_{calibrationPrefix}{c}" for c in "rgb"]
        ].values[None],
        cv2.COLOR_RGB2LAB,
    )[0]

    # Perform hierarchical clustering on primary and secondary lab colors.
    # Distances between two pollen are calculated by summing the distances of
    # primary and secondary colors. Color distances are calculated with
    # euclidean distance in lab color space.
    N = len(X_primary_lab)
    linkage_mode = "centroid"
    D_primary_condensed = pdist(X_primary_lab)

    if cluster_by == ClusterMode.PRIMARY_AND_SECONDARY:
        X_secondary_lab = cv2.cvtColor(
            df_pollen_colors[
                [f"secondary_{calibrationPrefix}{c}" for c in "rgb"]
            ].values[None],
            cv2.COLOR_RGB2LAB,
        )[0]
        D_secondary_condensed = pdist(X_secondary_lab)
        D_condensed = D_primary_condensed + D_secondary_condensed
    else:
        D_condensed = D_primary_condensed

    if len(D_condensed) == 0:
        # edge case for single pollen observation
        D_condensed = [1]
    linkage_matrix = linkage(D_condensed, linkage_mode)
    cluster_division = fcluster(
        Z=linkage_matrix,
        # this value was found to be a good threshold for splitting clusters.
        t=LAB_HIERARCHICAL_COLOR_DIFFERENCE_THRESHOLD * (
            1 if cluster_by == ClusterMode.PRIMARY else 2
        ),
        criterion="distance",
    )
    L.info(
        f"{N} pollen successfully divided into {cluster_division.max()} "
        "clusters."
    )

    return (
        linkage_matrix,
        cluster_division,
    )


def arbitrary_clusters(
    linkage_matrix: np.ndarray,
    df_pollen_colors: pd.DataFrame,
    useCalibratedColors: bool,
    cluster_by: ClusterMode = ClusterMode.PRIMARY,
    n_clusters: range = range(1, 20),
    outlier_threshold: float = 0.01
) -> List[Dict]:
    """Creates cluster color previews for arbitrary cluster counts.

    Args:
        linkage_matrix (np.ndarray): The pre calculated linkage matrix.
        df_pollen_colors (pd.DataFrame): The extracted pollen colors.
        useCalibratedColors (bool): If true, the calibrated colors are used
            for clustering. Make sure that the calibrated colors exist in the
            dataframe that is passed to this function.
        cluster_by (ClusterMode): If ClusterMode.PRIMARY, then secondary
            colors are ignored. If ClusterMode.PRIMARY_AND_SECONDARY, then
            secondary colors influence the clustering as much as the primary
            colors. Defaults to ClusterMode.PRIMARY.
        n_clusters (range, optional): The range of clusters per clustering.
            Defaults to range(1, 20).
        outlier_threshold (float, optional): Pollen that are smaller than
            this value (as percentage of all pollen detected) will be treated
            as outliers and are ignored for the calculation of biodiversity
            indices.

    Returns:
        Tuple[Dict, Dict]: A dict of clusters and a dict of diversity
        information.
    """
    if len(df_pollen_colors) <= 1:
        # nothing to cluster
        return dict(), dict()

    calibrationPrefix = "calib_" if useCalibratedColors else ""

    # Calculate for 20 clusters pollen-color-pixel-plots.
    primary_rgb_colors = df_pollen_colors[
        [f"primary_{calibrationPrefix}{c}" for c in "rgb"]
    ].values
    if cluster_by == ClusterMode.PRIMARY_AND_SECONDARY:
        secondary_rgb_colors = df_pollen_colors[
            [f"secondary_{calibrationPrefix}{c}" for c in "rgb"]
        ].values

    cluster_canvas = defaultdict(list)
    cluster_diversity = dict()
    for n_cluster in n_clusters:
        cluster_division = fcluster(
            Z=linkage_matrix, t=n_cluster, criterion="maxclust"
        )

        cluster_sizes = []
        n_pollen_excluding_outliers = 0
        for key in np.unique(cluster_division):
            cluster_size = np.where(cluster_division == key, 1, 0).sum()
            is_outlier = (
                cluster_size < len(df_pollen_colors) * outlier_threshold
            )
            if not is_outlier:
                n_pollen_excluding_outliers += cluster_size
            cluster_sizes.append((key, cluster_size, is_outlier))
        ordered_cluster_sizes = sorted(
            cluster_sizes,
            key=lambda kv: kv[1],
            reverse=True,
        )

        diversity_shannon = [(
            -cluster_size / n_pollen_excluding_outliers
            * np.log(cluster_size / n_pollen_excluding_outliers)
        ) for key, cluster_size, outlier in ordered_cluster_sizes
            if not outlier
        ]
        diversity_richness = len(diversity_shannon)
        diversity_shannon = np.sum(diversity_shannon)
        diversity_n = n_pollen_excluding_outliers
        diversity_avg_population_size = (
            diversity_n / diversity_richness
            if diversity_richness > 0
            else None
        )
        diversity_shannon_eveness = (
            diversity_shannon / np.log(diversity_richness)
            if np.log(diversity_richness) > 0
            else None
        )
        diversity = {
            "shannon": (
                float(diversity_shannon) if diversity_shannon else None
            ),
            "richness": (
                int(diversity_richness) if diversity_richness else None
            ),
            "n": int(diversity_n) if diversity_n else None,
            "avg_population_size": (
                float(diversity_avg_population_size)
                if diversity_avg_population_size
                else None
            ),
            "shannon_eveness": (
                float(diversity_shannon_eveness)
                if diversity_shannon_eveness
                else None
            )
        }
        cluster_diversity[n_cluster] = diversity

        for cluster_id, cluster_size, is_outlier in ordered_cluster_sizes:
            cluster_primary_colors = primary_rgb_colors[
                cluster_division == cluster_id
            ]
            if cluster_by == ClusterMode.PRIMARY_AND_SECONDARY:
                cluster_secondary_colors = secondary_rgb_colors[
                    cluster_division == cluster_id
                ]
            dim = int(np.ceil(len(cluster_primary_colors) ** 0.5))
            primary_canvas = np.ones((dim, dim, 3), np.uint8)
            if cluster_by == ClusterMode.PRIMARY_AND_SECONDARY:
                secondary_canvas = np.ones((dim, dim, 3), np.uint8)
            else:
                secondary_canvas = None
            for idx in range(len(cluster_primary_colors)):
                primary_canvas[idx // dim, idx % dim] = cluster_primary_colors[
                    idx
                ]
                if cluster_by == ClusterMode.PRIMARY_AND_SECONDARY:
                    secondary_canvas[
                        idx // dim, idx % dim
                    ] = cluster_secondary_colors[idx]

            cluster_canvas[n_cluster].append(
                {
                    "cluster_primary_canvas": primary_canvas,
                    "cluster_secondary_canvas": secondary_canvas,
                    "cluster_size": int(cluster_size),
                    "outlier": is_outlier
                }
            )
    L.info(f"{len(n_clusters)} arbitrary clusters formed.")
    return cluster_canvas, cluster_diversity
